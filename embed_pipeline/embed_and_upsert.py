"""
embed_and_upsert.py
====================
Direct ingestion script: seed_products_enriched.json -> bge-m3/CLIP -> Pinecone.

This implements the FINALIZED architecture (see Project_Master_Agentic_Context_Ledger.md,
Section 2, "Index vs. namespace clarified" + "Finalized Implementation for Embedding"):

  seed_products_enriched.json
     |- product text  -> bge-m3 -> Pinecone index "bge-m3-index" (single namespace)
     `- product image -> CLIP   -> Pinecone index "clip-index"   (single namespace)

  Query time: both branches run independently; results merged by product_id
  after retrieval. Vectors are NEVER combined or compared across indexes --
  that rule is enforced structurally here by using two separate Pinecone
  Index client objects with no shared code path between their upsert calls.

No LlamaIndex at ingestion (per Section 2 decision) -- this is the plain
~30-line-per-branch direct script that decision described, just written out
in full with error handling and resumability since 33,609 records is a real
batch job, not a one-off.

WHAT'S REAL:
  - Pinecone index creation/upsert calls: REAL (pinecone SDK).

Usage:
    python embed_and_upsert.py --dry-run                  # no API calls, validate shape
    python embed_and_upsert.py --dry-run --limit 20        # inspect a small sample
    python embed_and_upsert.py --limit 500                 # real run, capped
    python embed_and_upsert.py                              # real run, full dataset
    python embed_and_upsert.py --resume                     # continue after interruption

Requires PINECONE_API_KEY env var for real (non-dry-run) runs.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_PATH = PROJECT_ROOT / "seed_products_enriched.json"
PROGRESS_PATH = SCRIPT_DIR / "embed_progress.json"
SKIPPED_RECORDS_PATH = SCRIPT_DIR / "skipped_records.log"
PRODUCTS_MISSING_IMAGE_VECTOR_PATH = SCRIPT_DIR / "products_missing_image_vector.csv"

TEXT_INDEX_NAME = "bge-m3-index"
IMAGE_INDEX_NAME = "clip-index"
NAMESPACE = "products"   # same constant namespace in BOTH indexes -- text/image
                          # separation is handled entirely by which index object
                          # is called, never by namespace. See ledger note.

BGE_M3_DIM = 1024
CLIP_DIM = 512

BATCH_SIZE = 100
MAX_RETRIES = 1  # retry once, then log and skip -- same philosophy as the
                  # agent's tool-call error handling (graph.py), applied here
                  # to a batch ingestion context instead of a live request.


# ----------------------------------------------------------------------
# Text construction -- implements the finalized metadata split:
#   hard-filter-only (never in text): sizes, in_stock, fabric, sleeve_type
#     (as a stated preference), source_id/brand, price
#   semantic-text-only (never a filter): pattern, silhouette
#   both (filter AND text): occasion
# ----------------------------------------------------------------------

def build_embedding_text(product: dict) -> str:
    """
    Builds the string that gets embedded by bge-m3. Deliberately excludes
    sizes/in_stock/fabric/sleeve_type -- those are availability/hard-filter
    signal, not style signal, and would dilute the semantic content per the
    ledger's reasoning ("high-frequency, low-signal tokens that dilute the
    actual style signal the model should be keying on").
    """
    attrs = product.get("attributes", {}) or {}
    parts = [
        product.get("name", ""),
        product.get("category", ""),
        product.get("subcategory", ""),
        attrs.get("pattern", ""),
        attrs.get("silhouette", ""),
        attrs.get("occasion", ""),   # occasion: both filter AND text, per spec
        product.get("description", ""),
    ]
    return " | ".join(p for p in parts if p)


def build_metadata(product: dict) -> dict:
    """
    Metadata attached to BOTH the text and image vectors for this product
    (same metadata dict, two separate index upserts). This is where the
    hard-filter fields live: sizes, in_stock, fabric, sleeve_type, brand,
    price, plus occasion again (for the filter half of its dual role).

    Pinecone metadata values must be string/number/bool/list-of-string --
    no nested dicts, so sizes is normalized to a list here.
    """
    attrs = product.get("attributes", {}) or {}
    sizes_raw = product.get("sizes", "") or ""
    sizes_list = [s.strip() for s in sizes_raw.split(",") if s.strip()]

    metadata = {
        "product_url": product.get("product_url", ""),
        "image_url": product.get("image_url") or "",
        "name": product.get("name", ""),
        "brand": product.get("brand", ""),
        "source_id": product.get("source_id", ""),
        "category": product.get("category", ""),
        "subcategory": product.get("subcategory", ""),
        "price": float(product.get("price") or 0.0),
        "in_stock": bool(product.get("in_stock", 0)),
        "sizes": sizes_list,
        "fabric": product.get("fabric") or "",
        "sleeve_type": attrs.get("sleeve_type") or "",
        "neck_type": attrs.get("neck_type") or "",
        "occasion": attrs.get("occasion") or "",
        "curated_pairs": product.get("curated_pairs", []) or [],
    }
    return {key: "" if value is None else value for key, value in metadata.items()}


# ----------------------------------------------------------------------
# Embedding calls -- STUBBED, see module docstring
# ----------------------------------------------------------------------

def embed_text_bge_m3(text: str, dry_run: bool = False) -> list:
    """
    REAL (as of Sept 2026 provider decision): Qwen3-Embedding-8B on
    Fireworks serverless, resized to 1024-dim via the `dimensions` param.
    bge-m3 itself is confirmed unavailable serverless on both Fireworks and
    Nebius -- this replaces it while keeping the 1024-dim index spec intact.

    In --dry-run mode, falls back to a deterministic random vector so the
    pipeline can be validated without spending API calls.
    """
    if dry_run:
        import random
        random.seed(hash(text) % (2**31))
        return [random.uniform(-1, 1) for _ in range(BGE_M3_DIM)]

    import requests
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY not set")

    resp = requests.post(
        "https://api.fireworks.ai/inference/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "accounts/fireworks/models/qwen3-embedding-8b",
            "input": text,
            "dimensions": BGE_M3_DIM,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


# Module-level cache so the CLIP model loads once per process, not once per
# product -- loading it per-call would make the full 33,609-record run
# prohibitively slow.
_clip_model = None
_clip_load_seconds = None


def _get_clip_model():
    global _clip_model, _clip_load_seconds
    if _clip_model is None:
        from sentence_transformers import SentenceTransformer
        start = time.perf_counter()
        _clip_model = SentenceTransformer("clip-ViT-B-32")  # 512-dim, matches CLIP_DIM
        _clip_load_seconds = time.perf_counter() - start
        print(f"CLIP model loaded in {_clip_load_seconds:.2f}s")
    return _clip_model


def embed_image_clip(image_url: str, dry_run: bool = False) -> Optional[list]:
    """
    REAL (as of Sept 2026 provider decision): CLIP run locally via
    sentence-transformers (clip-ViT-B-32, 512-dim). Neither Fireworks nor
    Nebius offers a hosted image-embedding API -- both only serve text
    embedding models -- so this runs on the local machine instead, with no
    external provider or API cost.

    Returns None if image_url is missing, unfetchable, or not a valid image
    -- callers must skip the image branch for that record rather than
    upsert a meaningless vector, consistent with the retry-once-then-skip
    pattern used elsewhere in the pipeline.
    """
    if not image_url:
        return None

    if dry_run:
        import random
        random.seed(hash(image_url) % (2**31))
        return [random.uniform(-1, 1) for _ in range(CLIP_DIM)]

    import requests
    from PIL import Image
    from io import BytesIO

    try:
        resp = requests.get(image_url, timeout=15)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        print(f"  [clip] failed to fetch/open image {image_url}: {e}", file=sys.stderr)
        return None

    model = _get_clip_model()
    vec = model.encode(img)
    return vec.tolist()


# ----------------------------------------------------------------------
# Pinecone setup -- REAL
# ----------------------------------------------------------------------

def get_pinecone_indexes(dry_run: bool):
    """
    Returns (text_index, image_index) client objects, or (None, None) in
    dry-run mode. Creates the indexes if they don't exist yet.
    """
    if dry_run:
        return None, None

    from pinecone import Pinecone, ServerlessSpec

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("ERROR: PINECONE_API_KEY not set. Use --dry-run to test without it.",
              file=sys.stderr)
        sys.exit(1)

    pc = Pinecone(api_key=api_key)
    existing = {idx["name"] for idx in pc.list_indexes()}

    if TEXT_INDEX_NAME not in existing:
        print(f"Creating index '{TEXT_INDEX_NAME}' (dim={BGE_M3_DIM})...")
        pc.create_index(
            name=TEXT_INDEX_NAME, dimension=BGE_M3_DIM, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    if IMAGE_INDEX_NAME not in existing:
        print(f"Creating index '{IMAGE_INDEX_NAME}' (dim={CLIP_DIM})...")
        pc.create_index(
            name=IMAGE_INDEX_NAME, dimension=CLIP_DIM, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    return pc.Index(TEXT_INDEX_NAME), pc.Index(IMAGE_INDEX_NAME)


def log_skipped_records(index_label: str, refs: list, error: Exception) -> None:
    with open(SKIPPED_RECORDS_PATH, "a", encoding="utf-8") as f:
        for global_idx, pid in refs:
            f.write(f"{index_label}\t{global_idx}\t{pid}\t{error}\n")


def upsert_batch_with_retry(index, vectors: list, index_label: str, refs: list) -> int:
    """Upserts one batch, retrying once on failure. Returns count upserted (0 on failure)."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            index.upsert(vectors=vectors, namespace=NAMESPACE)
            return len(vectors)
        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"  [{index_label}] batch upsert failed (attempt {attempt + 1}), "
                      f"retrying: {e}", file=sys.stderr)
                time.sleep(2)
                continue
            print(f"  [{index_label}] batch upsert FAILED after retry, skipping "
                  f"{len(vectors)} records: {e}", file=sys.stderr)
            log_skipped_records(index_label, refs, e)
            return 0
    return 0


# ----------------------------------------------------------------------
# Resumability -- checkpoint the last completed product index so an
# interrupted run (network drop, rate limit, laptop sleep) doesn't require
# re-embedding everything from scratch.
# ----------------------------------------------------------------------

def load_progress() -> int:
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH) as f:
            return json.load(f).get("last_completed_index", -1)
    return -1


def save_progress(idx: int) -> None:
    with open(PROGRESS_PATH, "w") as f:
        json.dump({"last_completed_index": idx}, f)


def log_missing_image_vector(product_id: str, image_url: str, reason: str) -> None:
    needs_header = not PRODUCTS_MISSING_IMAGE_VECTOR_PATH.exists()
    with open(PRODUCTS_MISSING_IMAGE_VECTOR_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if needs_header:
            writer.writerow(["product_id", "image_url", "reason"])
        writer.writerow([product_id, image_url, reason])


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                         help="Build vectors/metadata and print samples, no Pinecone calls")
    parser.add_argument("--limit", type=int, default=None,
                         help="Cap the number of records processed")
    parser.add_argument("--resume", action="store_true",
                         help="Continue from the last checkpoint in embed_progress.json")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        products = json.load(f)
    print(f"Loaded {len(products)} products from {INPUT_PATH}")

    start_idx = 0
    if args.resume:
        start_idx = load_progress() + 1
        print(f"Resuming from index {start_idx} (checkpoint found).")

    targets = products[start_idx:]
    if args.limit:
        targets = targets[: args.limit]
    print(f"Processing {len(targets)} records "
          f"(indices {start_idx} to {start_idx + len(targets) - 1}).\n")

    text_index, image_index = get_pinecone_indexes(args.dry_run)

    text_batch, image_batch = [], []
    text_refs, image_refs = [], []
    text_upserted = image_upserted = image_skipped_no_photo = 0
    text_dim_seen = image_dim_seen = None

    def flush(text_batch, image_batch, text_refs, image_refs):
        nonlocal text_upserted, image_upserted
        if text_batch:
            if args.dry_run:
                text_upserted += len(text_batch)
            else:
                text_upserted += upsert_batch_with_retry(text_index, text_batch, "text", text_refs)
        if image_batch:
            if args.dry_run:
                image_upserted += len(image_batch)
            else:
                image_upserted += upsert_batch_with_retry(image_index, image_batch, "image", image_refs)
        return [], [], [], []

    for i, product in enumerate(targets):
        global_idx = start_idx + i
        pid = product.get("id", f"unknown-{global_idx}")

        text = build_embedding_text(product)
        metadata = build_metadata(product)

        text_vec = embed_text_bge_m3(text, dry_run=args.dry_run)
        if text_dim_seen is None:
            text_dim_seen = len(text_vec)
        text_batch.append({"id": pid, "values": text_vec, "metadata": metadata})
        text_refs.append((global_idx, pid))

        image_url = product.get("image_url", "") or ""
        image_vec = embed_image_clip(image_url, dry_run=args.dry_run)
        if image_vec is not None:
            if image_dim_seen is None:
                image_dim_seen = len(image_vec)
            image_batch.append({"id": pid, "values": image_vec, "metadata": metadata})
            image_refs.append((global_idx, pid))
        else:
            image_skipped_no_photo += 1
            if not args.dry_run:
                reason = "missing_url" if not image_url else "fetch_failed"
                log_missing_image_vector(pid, image_url, reason)

        if args.dry_run and i < 3:
            print(f"--- sample {i} ({pid}) ---")
            print(f"  embedding text: {text[:150]}...")
            print(f"  metadata: {metadata}")
            print(f"  text vector dim: {len(text_vec)}, "
                  f"image vector dim: {len(image_vec) if image_vec else 'SKIPPED (no image)'}\n")

        if len(text_batch) >= args.batch_size:
            text_batch, image_batch, text_refs, image_refs = flush(
                text_batch, image_batch, text_refs, image_refs)

        if not args.dry_run and (i + 1) % 500 == 0:
            save_progress(global_idx)
            print(f"  ...{i + 1}/{len(targets)} processed, checkpoint saved")

    flush(text_batch, image_batch, text_refs, image_refs)
    if not args.dry_run:
        save_progress(start_idx + len(targets) - 1)

    print(f"\n=== Done ===")
    print(f"Text vectors upserted:  {text_upserted}")
    print(f"Image vectors upserted: {image_upserted}")
    print("Image vectors skipped (missing url or fetch/decode failure): "
          f"{image_skipped_no_photo}")
    print(f"Text vector dim: {text_dim_seen}")
    print(f"Image vector dim: {image_dim_seen}")
    if _clip_load_seconds is not None:
        print(f"CLIP model load time: {_clip_load_seconds:.2f}s")
    if args.dry_run:
        print("\n(dry run -- nothing was actually sent to Pinecone)")


if __name__ == "__main__":
    main()
