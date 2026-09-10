"""
Tool implementations for the Digital Closet agent.

Everything in this file is a STUB right now so the graph in graph.py can run
and be demoed end-to-end before the embedding pipeline / Pinecone indexes /
Mem0 store exist. Every function below has a "REAL IMPLEMENTATION" comment
block showing exactly what to replace the stub body with once that piece is
built. Swapping a stub for the real call should never require touching
graph.py -- these functions are the seam.

Tool inventory (per Section 3 / Section 6 of the ledger):
  - search_brand_inventory   (read, autonomous)
  - get_user_preferences     (read, autonomous)
  - save_to_digital_closet   (write, requires human approval -- gated in graph.py,
                               not here)
  - vision_extract_attributes (not an MCP tool per se, but the vision model call)
  - build_complementary_query (orchestration LLM call)
  - rank_and_style             (orchestration LLM call)
"""

import json
import os
import random
import re
import time

import requests
from PIL import Image
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


class ToolError(Exception):
    """Raised by a tool stub to simulate a transient failure for the retry-once demo."""
    pass


_PINECONE_CLIENT = None
_CLIP_MODEL = None

TEXT_INDEX_NAME = "bge-m3-index"
IMAGE_INDEX_NAME = "clip-index"
NAMESPACE = "products"
TEXT_TOP_K = 15
IMAGE_TOP_K = 15
FINAL_TOP_K = 10
TEXT_ONLY_RANK_PENALTY = 0.05


def _get_pinecone_client() -> Pinecone:
    global _PINECONE_CLIENT
    if _PINECONE_CLIENT is None:
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise ToolError("PINECONE_API_KEY not set in environment")
        _PINECONE_CLIENT = Pinecone(api_key=api_key)
    return _PINECONE_CLIENT


def _get_clip_model() -> SentenceTransformer:
    global _CLIP_MODEL
    if _CLIP_MODEL is None:
        _CLIP_MODEL = SentenceTransformer("clip-ViT-B-32")
    return _CLIP_MODEL


def _embed_text_bge_m3(text: str) -> list:
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise ToolError("FIREWORKS_API_KEY not set in environment")
    resp = requests.post(
        "https://api.fireworks.ai/inference/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "accounts/fireworks/models/qwen3-embedding-8b",
            "input": text,
            "dimensions": 1024,
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def _embed_text_clip(text: str) -> list:
    model = _get_clip_model()
    vec = model.encode([text])[0]
    return vec.tolist()


def _embed_image_clip(image_path: str) -> list:
    try:
        image = Image.open(image_path).convert("RGB")
    except (FileNotFoundError, OSError) as e:
        raise ToolError(f"search_brand_inventory: couldn't open image_path {image_path!r}: {e}")
    model = _get_clip_model()
    vec = model.encode(image)
    return vec.tolist()


def _build_metadata_filter(query: dict) -> dict:
    # occasion removed as a hard filter -- live Pinecone metadata stores raw
    # per-brand tag strings (e.g. "office outfits"), which won't reliably
    # $eq-match the LLM's normalized enum (casual/formal/work/party).
    # Occasion intent rides in text_query wording instead, same as category.
    return {"in_stock": {"$eq": True}}


def _match_id(match):
    return match.id if hasattr(match, "id") else match["id"]


def _match_score(match):
    return match.score if hasattr(match, "score") else match["score"]


def _match_metadata(match):
    meta = getattr(match, "metadata", None)
    if meta is None and isinstance(match, dict):
        meta = match.get("metadata")
    return meta or {}


def _response_matches(resp):
    matches = getattr(resp, "matches", None)
    if matches is None and isinstance(resp, dict):
        matches = resp.get("matches", [])
    return matches or []


def _to_candidate(product_id: str, metadata: dict, score: float) -> dict:
    has_image = bool(metadata.get("has_image_vector", True))
    return {
        "id": product_id,
        "name": metadata.get("name", ""),
        "brand": metadata.get("brand", ""),
        "price": metadata.get("price", 0.0),
        "product_url": metadata.get("product_url", ""),
        "image_url": metadata.get("image_url", ""),
        "category": metadata.get("category", ""),
        "score": score,
        "has_image_vector": has_image,
        "display_mode": "photo" if has_image else "text_only",
    }


# ----------------------------------------------------------------------
# Vision extraction (Qwen2.5-VL originally, now GLM-5.3-Flash per ledger)
# ----------------------------------------------------------------------

def vision_extract_attributes(image_path: str, simulate_failure: bool = False) -> dict:
    """
    STUB. Returns a plausible attribute set + a confidence score.

    REAL IMPLEMENTATION:
      Call GLM-5.3-Flash via Fireworks or Nebius (see backfill_attributes.py's
      call_vision_model / call_vision_model_nebius for the exact request shape
      and the re.search(r'{.*}', content, re.DOTALL) JSON-extraction fix).
      Parse the JSON response into {neck_type, sleeve_type, pattern,
      silhouette}, and additionally derive a confidence score (e.g. from
      logprobs if available, or from how many fields the model returned vs.
      how many it was asked for).
    """
    if simulate_failure:
        raise ToolError("Vision model request failed")

    time.sleep(0.1)  # pretend this is a network call
    stub_attributes = {
        "neck_type": "v-neck",
        "sleeve_type": "sleeveless",
        "pattern": "solid",
        "silhouette": "fitted",
        "primary_color": "black",
    }
    # Randomized confidence so the demo can show BOTH branches (confident /
    # low-confidence re-upload) across a few runs.
    confidence = round(random.uniform(0.55, 0.98), 2)
    return {"attributes": stub_attributes, "confidence": confidence}


# ----------------------------------------------------------------------
# Orchestration LLM: query construction
# ----------------------------------------------------------------------

def build_complementary_query(attributes: dict, broaden: bool = False,
                              simulate_failure: bool = False) -> dict:
    if simulate_failure:
        raise ToolError("Orchestration LLM request failed")
    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        raise ToolError("NEBIUS_API_KEY not set in environment")

    system_prompt = (
        "You are a fashion stylist assistant. Given attributes extracted "
        "from a photo of ONE clothing item a user already owns, propose a "
        "search query for a DIFFERENT, complementary item that would pair "
        "well with it (e.g. a top's complementary item is a bottom, skirt, "
        "or outerwear piece -- never another top; a bottom's complementary "
        "item is a top or outerwear piece -- never another bottom). "
        "Respond with ONLY a JSON object with these exact keys: "
        "target_category (a short lowercase phrase naming the complementary "
        "category, e.g. \"bottoms\", \"outerwear\", \"accessories\"), "
        "compatible_colors (list of 2-6 color strings that pair well with "
        "the item's primary_color), occasion (one of: casual, formal, work, "
        "party), text_query (a short natural-language description of the "
        "ideal complementary piece for that occasion, written the way a "
        "shopper would search for it, e.g. \"high-waisted wide leg trousers "
        "in neutral tones for the office\"). "
        "No reasoning, no explanation, no markdown, JSON only."
    )
    user_prompt = f"Item attributes: {json.dumps(attributes)}"
    if broaden:
        user_prompt += (
            "\n\nThe first search came back with too few results. Broaden "
            "the query: widen compatible_colors to include more neutrals "
            "(black, white, beige, brown, grey, navy), and make text_query "
            "less restrictive so it isn't limited to one narrow category."
        )

    resp = requests.post(
        "https://api.studio.nebius.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "meta-llama/Llama-3.3-70B-Instruct",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 400,
            "temperature": 0.3,
        },
        timeout=20,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ToolError(f"build_complementary_query: no JSON found in LLM response: {content!r}")
    query = json.loads(match.group(0))

    query.setdefault("target_category", None if broaden else "bottoms")
    query.setdefault("compatible_colors", [])
    query.setdefault("occasion", attributes.get("occasion", "casual"))
    query.setdefault("text_query", "")
    if broaden:
        query["target_category"] = None
    return query


# ----------------------------------------------------------------------
# search_brand_inventory (read tool -> dual Pinecone query)
# ----------------------------------------------------------------------

_search_call_count = {"n": 0}  # module-level, only used to drive the one-time
                                # simulated failure in demo.py -- not real state


def search_brand_inventory(query: dict, image_path: str = None,
                            simulate_failure: bool = False,
                            simulate_sparse: bool = False) -> list:
    if simulate_failure:
        _search_call_count["n"] += 1
        if _search_call_count["n"] == 1:
            raise ToolError("Pinecone query timed out")

    if simulate_sparse == "zero":
        return []

    pc = _get_pinecone_client()
    text_index = pc.Index(TEXT_INDEX_NAME)
    image_index = pc.Index(IMAGE_INDEX_NAME)

    text_query_str = (query.get("text_query") or "").strip()
    if not text_query_str:
        raise ToolError("search_brand_inventory: empty text_query in query dict")

    metadata_filter = _build_metadata_filter(query)
    text_vec = _embed_text_bge_m3(text_query_str)

    if image_path:
        clip_vec = _embed_image_clip(image_path)
    else:
        clip_vec = _embed_text_clip(text_query_str)

    text_resp = text_index.query(
        vector=text_vec, top_k=TEXT_TOP_K, namespace=NAMESPACE,
        filter=metadata_filter, include_metadata=True,
    )
    image_resp = image_index.query(
        vector=clip_vec, top_k=IMAGE_TOP_K, namespace=NAMESPACE,
        filter=metadata_filter, include_metadata=True,
    )

    merged = {}
    for match in _response_matches(text_resp):
        mid = _match_id(match)
        merged[mid] = _to_candidate(mid, _match_metadata(match), _match_score(match))

    for match in _response_matches(image_resp):
        mid = _match_id(match)
        score = _match_score(match)
        if mid in merged:
            merged[mid]["score"] = max(merged[mid]["score"], score)
        else:
            merged[mid] = _to_candidate(mid, _match_metadata(match), score)

    candidates = list(merged.values())
    candidates.sort(
        key=lambda c: c["score"] - (TEXT_ONLY_RANK_PENALTY if c["display_mode"] == "text_only" else 0.0),
        reverse=True,
    )

    if simulate_sparse:
        return candidates[:1]

    return candidates[:FINAL_TOP_K]


# ----------------------------------------------------------------------
# get_user_preferences (read tool -> Mem0)
# ----------------------------------------------------------------------

def get_user_preferences(user_id: str) -> dict:
    """
    STUB. REAL IMPLEMENTATION: fetch the Mem0 user profile (disliked colors/
    brands, past saved closet items, style preferences) for `user_id`.
    """
    return {
        "disliked_colors": ["orange"],
        "preferred_brands": ["reddress.com", "bohme.com"],
        "budget_max": 150.0,
    }


# ----------------------------------------------------------------------
# Orchestration LLM: ranking + styling copy
# ----------------------------------------------------------------------

def rank_and_style(candidates: list, preferences: dict) -> tuple[list, str]:
    """
    STUB. REAL IMPLEMENTATION: call Llama-3.3-70B with candidates +
    preferences, have it drop/deprioritize disliked colors/brands and
    over-budget items, then return a ranked list plus a short styling note.
    """
    ranked = [c for c in candidates if c["price"] <= preferences.get("budget_max", 9999)]
    ranked.sort(key=lambda c: c["id"] not in [], reverse=False)  # placeholder, real: LLM-ranked
    note = ("These pair well with a fitted black sleeveless top -- the wide-leg "
            "pants add contrast, the skort keeps it casual.")
    return ranked, note


# ----------------------------------------------------------------------
# save_to_digital_closet (WRITE tool -- human approval enforced in graph.py,
# never here)
# ----------------------------------------------------------------------

def save_to_digital_closet(user_id: str, item_id: str) -> bool:
    """
    STUB. REAL IMPLEMENTATION: INSERT into PostgreSQL via the MCP server,
    the user's saved closet items table. This function assumes approval has
    ALREADY happened -- graph.py must never call this without having passed
    through the interrupt_before gate.
    """
    return True
