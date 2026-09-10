"""
One-off repair for products whose text vectors exist but CLIP image vectors
were skipped because the original image URL failed during ingestion.

This updates only the matching IDs in clip-index; it does not touch text
vectors or rerun the full ingestion pipeline.
"""

import json
import os
import sys
from pathlib import Path

from pinecone import Pinecone

from embed_and_upsert import (
    IMAGE_INDEX_NAME,
    INPUT_PATH,
    NAMESPACE,
    build_metadata,
    embed_image_clip,
)

REPAIRED_IMAGE_URLS = {
    "meshki.us:7742137040992": (
        "https://cdn.shopify.com/s/files/1/0017/7920/4211/files/"
        "06_PIA_DRESS_PERIWINKLE_0007.jpg?v=1787025879"
    ),
    "naturallife.com:8710332645549": (
        "https://cdn.shopify.com/s/files/1/0409/9656/9251/files/"
        "DSC09859_20copy.webp?v=1788563881"
    ),
    "reddress.com:8200695939261": (
        "https://cdn.shopify.com/s/files/1/1708/7943/files/"
        "Bronze-Pleated-Asymmetric-One-Shoulder-Max-Dress.jpg?v=1788446538"
    ),
}


def main() -> None:
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("ERROR: PINECONE_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    products = json.loads(Path(INPUT_PATH).read_text(encoding="utf-8"))
    by_id = {product.get("id"): product for product in products}

    pc = Pinecone(api_key=api_key)
    image_index = pc.Index(IMAGE_INDEX_NAME)

    vectors = []
    for product_id, expected_url in REPAIRED_IMAGE_URLS.items():
        product = by_id[product_id]
        actual_url = product.get("image_url") or ""
        if actual_url != expected_url:
            raise RuntimeError(
                f"{product_id} image_url mismatch: expected {expected_url}, got {actual_url}"
            )

        image_vec = embed_image_clip(actual_url)
        if image_vec is None:
            raise RuntimeError(f"CLIP embedding failed for {product_id}: {actual_url}")

        vectors.append({
            "id": product_id,
            "values": image_vec,
            "metadata": build_metadata(product),
        })
        print(f"Prepared CLIP vector for {product_id}")

    image_index.upsert(vectors=vectors, namespace=NAMESPACE)
    print(f"Upserted {len(vectors)} repaired image vectors into {IMAGE_INDEX_NAME}/{NAMESPACE}")


if __name__ == "__main__":
    main()
