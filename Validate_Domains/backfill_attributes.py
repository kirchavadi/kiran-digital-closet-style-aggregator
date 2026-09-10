"""
Backfill garment attribute metadata (neck, sleeve, pattern, occasion, silhouette)
==================================================================================
Runs AFTER color/fabric backfill, BEFORE the bge-m3/CLIP embedding step.

Two passes, same pattern as backfill_seed_products.py:
  Pass 1 (free, instant):  parse per-store tag conventions where they exist
                            (bohme, petalandpup use neck:/sleeve:/occ:/style:/print:)
  Pass 2 (paid, optional):  Qwen2.5-VL vision fallback via Fireworks, for records
                            still missing attributes after Pass 1 (meshki, ohpolly,
                            saboskirt, reddress mostly lack structured tags)

Also extracts `curated_pairs`: product IDs referenced in `pair:` and `setitem-`
tags. These are brand-editorial "goes with" relationships -- deferred Neo4j
value captured now as plain metadata, no graph DB required for v1.

Usage:
    python backfill_attributes.py                     # tag-parsing pass only
    python backfill_attributes.py --run-vision         # also run vision fallback
    python backfill_attributes.py --run-vision --limit 50   # test on 50 records first

Requires: FIREWORKS_API_KEY env var if --run-vision is used.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

INPUT_PATH = "../seed_products_final.json"
OUTPUT_PATH = "../seed_products_enriched.json"

# ----------------------------------------------------------------------
# Controlled vocabularies -- keep these tight. An LLM or regex match that
# falls outside these sets gets dropped rather than polluting metadata
# with one-off free text that can't be used as a filter.
# ----------------------------------------------------------------------

NECK_TYPES = [
    "v-neck", "round neck", "boat neck", "halter", "off shoulder",
    "sweetheart", "collared", "turtleneck", "cowl", "square neck",
    "high neck", "scoop neck", "bateau",
]

SLEEVE_TYPES = [
    "sleeveless", "short sleeve", "long sleeve", "3/4 sleeve",
    "cap sleeve", "puff sleeve", "strapless",
]

OCCASIONS = [
    "office outfits", "casual", "party", "wedding guest", "vacation",
    "date night", "formal", "bridal", "festival", "resort wear",
    "everyday", "eventwear",
]

PATTERNS = [
    "floral", "striped", "solid", "polka dot", "animal print",
    "plaid", "checked", "geometric", "abstract",
]

SILHOUETTES = [
    "a-line", "bodycon", "wrap", "cut out", "fitted", "relaxed",
    "oversized", "structured", "flowy",
]

# ----------------------------------------------------------------------
# Pass 1: per-store tag-convention parsing
# ----------------------------------------------------------------------

def _match_from_vocab(value, vocab):
    """Case-insensitive match of a raw tag value against a controlled vocab."""
    value_l = value.strip().lower()
    for v in vocab:
        if v == value_l or v in value_l:
            return v
    return None


def parse_tags_for_attributes(tags_str):
    """
    Parses a raw tags string like:
      'category:Dresses, neck:Sweetheart, sleeve:Sleeveless, occ:Party, ...'
    Handles the prefixed conventions used by bohme.com and petalandpup.com.
    Returns a dict with whichever fields were found (missing keys omitted).
    """
    found = {}
    if not tags_str:
        return found

    tags = [t.strip() for t in tags_str.split(",")]

    prefix_map = {
        "neck:": ("neck_type", NECK_TYPES),
        "sleeve:": ("sleeve_type", SLEEVE_TYPES),
        "occ:": ("occasion", OCCASIONS),
        "print:": ("pattern", PATTERNS),
        "style:": ("silhouette", SILHOUETTES),
    }

    for tag in tags:
        tag_l = tag.lower()
        for prefix, (field, vocab) in prefix_map.items():
            if tag_l.startswith(prefix):
                raw_value = tag[len(prefix):]
                matched = _match_from_vocab(raw_value, vocab)
                if matched and field not in found:
                    found[field] = matched

    return found


def parse_bare_category_tags(tags_str):
    """
    Catches occasion/pattern/silhouette signal sitting in unprefixed category
    tags (e.g. 'category:Festival', 'category:Bridal') that the prefix-based
    parser above doesn't recognize. Deliberately excludes neck/sleeve -- no
    bare-tag evidence for those fields in this dataset, so widening there
    would risk false positives.
    """
    found = {}
    if not tags_str:
        return found

    tags = [t.strip() for t in tags_str.split(",")]
    for tag in tags:
        tag_l = tag.lower()
        # Strip known non-attribute prefixes so we're matching the bare value
        for strip_prefix in ["category:", "colour:", "sticker:", "model:"]:
            if tag_l.startswith(strip_prefix):
                tag_l = tag_l[len(strip_prefix):]
                break

        for field, vocab in [
            ("occasion", OCCASIONS), ("pattern", PATTERNS), ("silhouette", SILHOUETTES),
        ]:
            if field in found:
                continue
            matched = _match_from_vocab(tag_l, vocab)
            if matched:
                found[field] = matched

    return found


def parse_description_for_attributes(description):
    """
    Fallback keyword scan for stores with no structured tags at all
    (meshki, ohpolly, saboskirt). Same approach as the fabric backfill's
    regex/keyword pass -- lower recall than tags, still genuine signal.
    """
    found = {}
    if not description:
        return found

    desc_l = description.lower()

    for field, vocab in [
        ("neck_type", NECK_TYPES),
        ("sleeve_type", SLEEVE_TYPES),
        ("pattern", PATTERNS),
        ("silhouette", SILHOUETTES),
    ]:
        for v in vocab:
            if v in desc_l:
                found[field] = v
                break

    return found


def parse_curated_pairs(tags_str):
    """
    Extracts brand-editorial 'goes with' relationships from:
      pair:mai-sling-back-thong-heel-white
      setitem-19863-20651
    These are curated pairings the brand already decided -- captured as plain
    metadata now, no Neo4j needed for v1. A future graph-enrichment pass could
    traverse these as edges without re-scraping.
    """
    pairs = []
    if not tags_str:
        return pairs

    for tag in [t.strip() for t in tags_str.split(",")]:
        if tag.lower().startswith("pair:"):
            pairs.append(tag[len("pair:"):])
        elif tag.lower().startswith("setitem-"):
            pairs.append(tag)

    return pairs


# ----------------------------------------------------------------------
# Pass 2: Qwen2.5-VL vision fallback (optional, paid, off by default)
# ----------------------------------------------------------------------

VISION_PROMPT = """You are tagging a fashion product photo. Look at the image
and return ONLY a JSON object (no markdown, no preamble) with any of these
fields you can confidently determine. Omit a field entirely if unsure -- do
not guess.

{
  "neck_type": one of """ + json.dumps(NECK_TYPES) + """,
  "sleeve_type": one of """ + json.dumps(SLEEVE_TYPES) + """,
  "pattern": one of """ + json.dumps(PATTERNS) + """,
  "silhouette": one of """ + json.dumps(SILHOUETTES) + """
}
"""


def call_vision_model(image_url, api_key, max_retries=2):
    """
    Calls Qwen2.5-VL via Fireworks for a single product image.
    Returns a dict of extracted attributes, or {} on failure.

    NOTE: requires the `requests` package and network access to
    api.fireworks.ai. Kept as a plain HTTP call (no SDK dependency)
    so this script runs anywhere without extra installs.
    """
    import requests

    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "accounts/fireworks/models/glm-5p3-flash",
        "max_tokens": 1200,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    }

    for attempt in range(max_retries + 1):
        try:
            print(f"[DEBUG] max_tokens being sent: {payload['max_tokens']}")
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            response_json = resp.json()
            choice = response_json["choices"][0]
            content = choice["message"].get("content")
            if not content:
                print(f"  [vision] finish_reason: {choice.get('finish_reason')}", file=sys.stderr)
                print(f"  [vision] raw response: {response_json}", file=sys.stderr)
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON object found in response: {content[:200]}")
            parsed = json.loads(match.group(0))
            # Validate against controlled vocab -- drop anything off-list
            clean = {}
            for field, vocab in [
                ("neck_type", NECK_TYPES), ("sleeve_type", SLEEVE_TYPES),
                ("pattern", PATTERNS), ("silhouette", SILHOUETTES),
            ]:
                val = parsed.get(field, "").strip().lower() if parsed.get(field) else None
                if val in vocab:
                    clean[field] = val
            return clean
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            print(f"  [vision] failed for {image_url}: {e}", file=sys.stderr)
            return {}


def call_vision_model_nebius(image_url, api_key, max_retries=2):
    import requests

    url = "https://api.tokenfactory.nebius.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "zai-org/GLM-5.3-Flash",
        "max_tokens": 1200,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    }

    for attempt in range(max_retries + 1):
        try:
            print(f"[DEBUG] max_tokens being sent: {payload['max_tokens']}")
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            response_json = resp.json()
            choice = response_json["choices"][0]
            content = choice["message"].get("content")
            if not content:
                print(f"  [vision-nebius] finish_reason: {choice.get('finish_reason')}", file=sys.stderr)
                print(f"  [vision-nebius] raw response: {response_json}", file=sys.stderr)
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON object found in response: {content[:200]}")
            parsed = json.loads(match.group(0))
            clean = {}
            for field, vocab in [
                ("neck_type", NECK_TYPES), ("sleeve_type", SLEEVE_TYPES),
                ("pattern", PATTERNS), ("silhouette", SILHOUETTES),
            ]:
                val = parsed.get(field, "").strip().lower() if parsed.get(field) else None
                if val in vocab:
                    clean[field] = val
            return clean
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            print(f"  [vision-nebius] failed for {image_url}: {e}", file=sys.stderr)
            return {}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-vision", action="store_true",
                         help="Run Qwen2.5-VL fallback pass for records with no tag signal")
    parser.add_argument("--provider", choices=["fireworks", "nebius"], default="fireworks",
                         help="Which vision API to use for --run-vision (default: fireworks)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Cap vision calls to N records (for testing spend before a full run)")
    args = parser.parse_args()

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        products = json.load(f)

    print(f"Loaded {len(products)} products from {INPUT_PATH}\n")

    field_names = ["neck_type", "sleeve_type", "occasion", "pattern", "silhouette"]
    missing_before = Counter()
    missing_after_pass1 = Counter()
    vision_calls = 0
    vision_recovered = Counter()

    # --- Pass 1: tags, then description keyword fallback ---
    needs_vision = []

    for p in products:
        attrs = {}
        attrs.update(parse_tags_for_attributes(p.get("tags", "")))

        for f, v in parse_bare_category_tags(p.get("tags", "")).items():
            if f not in attrs:
                attrs[f] = v

        still_missing = [f for f in field_names if f not in attrs]
        if still_missing:
            desc_attrs = parse_description_for_attributes(p.get("description", ""))
            for f in still_missing:
                if f in desc_attrs:
                    attrs[f] = desc_attrs[f]

        p["curated_pairs"] = parse_curated_pairs(p.get("tags", ""))

        for f in field_names:
            if f not in attrs:
                missing_before[f] += 1

        # Vision candidates: no neck/sleeve/pattern/silhouette signal at all,
        # since occasion is rarely visually determinable and skipped for vision.
        visual_fields = ["neck_type", "sleeve_type", "pattern", "silhouette"]
        if all(f not in attrs for f in visual_fields) and p.get("image_url"):
            needs_vision.append(p)
        else:
            for f in visual_fields:
                if f not in attrs:
                    missing_after_pass1[f] += 1

        p["attributes"] = attrs

    print("=== Pass 1 (tags + description) results ===")
    for f in field_names:
        recovered = len(products) - missing_before[f]
        print(f"  {f:12s}: {recovered:5d}/{len(products)} recovered "
              f"({100*recovered/len(products):.1f}%)")
    print(f"\n{len(needs_vision)} records have zero visual-attribute signal "
          f"and would need the vision pass.\n")

    # --- Pass 2: vision fallback (optional) ---
    if args.run_vision:
        if args.provider == "nebius":
            api_key = os.environ.get("NEBIUS_API_KEY")
            vision_fn = call_vision_model_nebius
            provider_label = "Nebius"
        else:
            api_key = os.environ.get("FIREWORKS_API_KEY")
            vision_fn = call_vision_model
            provider_label = "Fireworks"

        if not api_key:
            print(f"ERROR: --run-vision with --provider {args.provider} requires "
                  f"{'NEBIUS_API_KEY' if args.provider == 'nebius' else 'FIREWORKS_API_KEY'} to be set.",
                  file=sys.stderr)
            sys.exit(1)

        targets = needs_vision[: args.limit] if args.limit else needs_vision
        print(f"Running {provider_label} vision fallback on {len(targets)} records "
              f"(of {len(needs_vision)} eligible)...\n")

        for i, p in enumerate(targets, 1):
            result = vision_fn(p["image_url"], api_key)
            vision_calls += 1
            for f, v in result.items():
                p["attributes"][f] = v
                vision_recovered[f] += 1
            if i % 25 == 0:
                print(f"  ...{i}/{len(targets)} processed")

        print(f"\n=== Vision pass results ({vision_calls} calls) ===")
        for f in ["neck_type", "sleeve_type", "pattern", "silhouette"]:
            print(f"  {f:12s}: {vision_recovered[f]} additionally recovered")
    else:
        print("Skipping vision pass (pass --run-vision to enable). "
              "Tag/description-only attributes will be saved.\n")

    # --- Final stats + curated pairs summary ---
    with_pairs = sum(1 for p in products if p.get("curated_pairs"))
    print(f"\n{with_pairs} products have at least one curated_pairs entry "
          f"(brand-editorial pair:/setitem- tags).")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)

    print(f"\nSaved enriched dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
