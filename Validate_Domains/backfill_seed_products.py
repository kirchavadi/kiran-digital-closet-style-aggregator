"""
Backfill missing `colors` and `fabric` fields in seed_products.json (v2)
=========================================================================
Tuned against real samples pulled from all 7 stores (saboskirt, petalandpup,
meshki, bohme, naturallife, ohpolly, reddress). Each store encodes color and
fabric differently -- this script tries multiple sources per field, in order
of confidence, and only falls back to weaker signals when stronger ones are
missing.

COLOR sources, in priority order:
  1. tags: primary-color:X   (saboskirt)
  2. tags: colour:X          (petalandpup -- British spelling)
  3. tags: Color_X / ColorFamily_X / Dress_Color:X   (bohme)
  4. tags: COLOR-X           (reddress -- hyphenated, uppercase)
  5. product name suffix " - Color" or " in Color"   (meshki, ohpolly --
     these stores put zero color info in tags, only in the title)

FABRIC sources, in priority order:
  1. tags: fabric:X          (petalandpup -- explicit, most reliable)
  2. description: "NN% Material" composition call-outs (bohme, reddress)
  3. description: fabric keyword scan (last resort, lower confidence)

Usage:
    python backfill_seed_products.py seed_products.json

Writes seed_products_backfilled.json + prints per-store before/after stats.
"""

import json
import re
import sys
from collections import defaultdict

# ----------------------------------------------------------------------
# COLOR: tag-based patterns (highest confidence)
# ----------------------------------------------------------------------
COLOR_TAG_PATTERNS = [
    r'primary-color:([^,]+)',       # saboskirt
    r'colour:([^,]+)',              # petalandpup
    r'\bColor_([^,]+)',             # bohme
    r'ColorFamily_([^,]+)',         # bohme
    r'Dress_Color:([^,]+)',         # bohme
    r'COLOR-([^,]+)',               # reddress
    r'swatch:([^,]+)',              # naturallife
]

# Generic/non-color trailing words to reject when falling back to product
# title parsing (meshki / ohpolly) -- avoids grabbing "FINAL SALE", set
# names, or garment type words that happen to follow a dash.
TITLE_COLOR_BLOCKLIST = {
    "final sale", "sale", "new", "set", "sample", "bundle", "restock",
    "back in stock", "1", "2", "3", "top", "dress", "pants", "shorts",
    "skirt", "sweater", "jacket", "vest", "with hardware"
}

TITLE_DASH_PATTERN = re.compile(r'\s-\s([A-Za-z][A-Za-z\s]{2,24})$')
TITLE_IN_PATTERN = re.compile(r'\bin\s([A-Za-z][A-Za-z\s]{2,24})$')


def extract_color_from_tags(tags):
    found = []
    for pattern in COLOR_TAG_PATTERNS:
        for match in re.finditer(pattern, tags):
            val = match.group(1).strip()
            if val and val.lower() not in [c.lower() for c in found]:
                found.append(val)
    return found


def extract_color_from_title(name):
    # Strip common non-color trailing modifiers first
    cleaned = re.sub(r'\s*-\s*(FINAL SALE|SALE)\s*$', '', name, flags=re.IGNORECASE)

    for pattern in (TITLE_DASH_PATTERN, TITLE_IN_PATTERN):
        match = pattern.search(cleaned)
        if match:
            candidate = match.group(1).strip()
            if candidate.lower() not in TITLE_COLOR_BLOCKLIST and not any(ch.isdigit() for ch in candidate):
                return [candidate]
    return []


def backfill_color(product):
    if product.get("colors", "[]") not in ("[]", "", None):
        return product  # already has color data

    tags = product.get("tags", "") or ""
    found = extract_color_from_tags(tags)

    if not found:
        name = product.get("name", "") or ""
        found = extract_color_from_title(name)

    if found:
        product["colors"] = json.dumps(found)
    return product


# ----------------------------------------------------------------------
# FABRIC: tag-based pattern (highest confidence)
# ----------------------------------------------------------------------
FABRIC_TAG_PATTERN = re.compile(r'\bfabric:([^,]+)', re.IGNORECASE)

# Percentage composition, e.g. "54% Tencel", "3% Spandex". Single-word
# capture only -- avoids swallowing "&", "Unlined", "Lining:" etc. that
# follow in the same sentence.
FABRIC_PCT_PATTERN = re.compile(r'(\d{1,3})%\s*([A-Za-z][a-zA-Z\-]{2,20})\b')

FABRIC_KEYWORDS = [
    "cotton", "linen", "silk", "polyester", "rayon", "viscose", "nylon",
    "spandex", "elastane", "tencel", "denim", "velvet", "lace", "chiffon",
    "satin", "wool", "cashmere", "leather", "suede", "modal", "acrylic",
    "jersey", "corduroy", "fleece", "flannel"
]


def backfill_fabric(product):
    if product.get("fabric"):
        return product  # already populated

    tags = product.get("tags", "") or ""
    tag_matches = FABRIC_TAG_PATTERN.findall(tags)
    if tag_matches:
        cleaned = [m.strip() for m in tag_matches]
        # de-dupe, preserve order
        seen = []
        for c in cleaned:
            if c.lower() not in [s.lower() for s in seen]:
                seen.append(c)
        product["fabric"] = ", ".join(seen)
        return product

    desc = product.get("description", "") or ""
    pct_matches = FABRIC_PCT_PATTERN.findall(desc)
    if pct_matches:
        parts = [f"{pct}% {name}" for pct, name in pct_matches]
        product["fabric"] = ", ".join(parts)
        return product

    desc_lower = desc.lower()
    found_keywords = sorted(set(kw.capitalize() for kw in FABRIC_KEYWORDS if kw in desc_lower))
    if found_keywords:
        product["fabric"] = ", ".join(found_keywords)

    return product


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python backfill_seed_products.py <path_to_seed_products.json>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = input_path.replace(".json", "_backfilled.json")

    print(f"Loading {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        products = json.load(f)

    print(f"Loaded {len(products)} products.\n")

    def snapshot():
        stats = defaultdict(lambda: {"total": 0, "empty_color": 0, "empty_fabric": 0})
        for p in products:
            store = p.get("source_id", "unknown")
            stats[store]["total"] += 1
            if p.get("colors", "[]") in ("[]", "", None):
                stats[store]["empty_color"] += 1
            if not p.get("fabric"):
                stats[store]["empty_fabric"] += 1
        return stats

    before = snapshot()

    for p in products:
        backfill_color(p)
        backfill_fabric(p)

    after = snapshot()

    print(f"{'Store':<20}{'Total':>8}   {'Color missing (before -> after)':<34}{'Fabric missing (before -> after)'}")
    for store in sorted(before.keys()):
        b, a = before[store], after[store]
        color_str = f"{b['empty_color']:>4} -> {a['empty_color']:<4}"
        fabric_str = f"{b['empty_fabric']:>4} -> {a['empty_fabric']:<4}"
        print(f"{store:<20}{b['total']:>8}   {color_str:<34}{fabric_str}")

    total_before_color = sum(s["empty_color"] for s in before.values())
    total_after_color = sum(s["empty_color"] for s in after.values())
    total_before_fabric = sum(s["empty_fabric"] for s in before.values())
    total_after_fabric = sum(s["empty_fabric"] for s in after.values())

    print(f"\nTOTAL color gaps filled:  {total_before_color - total_after_color} "
          f"({total_before_color} -> {total_after_color})")
    print(f"TOTAL fabric gaps filled: {total_before_fabric - total_after_fabric} "
          f"({total_before_fabric} -> {total_after_fabric})")

    print(f"\nWriting backfilled file to {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
