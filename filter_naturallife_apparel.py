"""
Filter non-apparel naturallife.com records out of seed_products_backfilled.json
=================================================================================
naturallife.com is a lifestyle/gift brand, not pure apparel -- its catalog
mixes clothing with mugs, camping gear, gift novelties, etc. This script
removes the non-apparel subcategories from naturallife.com ONLY. All other
6 stores are untouched (they're pure apparel/accessories already).

Decision was made by spot-checking actual product names per subcategory:

  KEEP (naturallife.com): Tops, Tees, Sweaters, Sweatshirts, Bottoms,
    Dresses, Jumpsuits, Pajamas & Intimates, Kimonos & Coverups, Outerwear,
    Socks & Slippers, Wearable Accessories, Bags & Pouches, Hair

  DROP (naturallife.com): Kitchen, Car, Bath, Home Décor, Bedroom,
    Stationery & Planners, Porch & Backyard, Heartfelt Gift with Words,
    Gift Card, Marketing, Dummy, Beach/Camping/Outdoor,
    Unique & Fun Treasures, Happy Bags & Totes, Blankets

Usage:
    python filter_naturallife_apparel.py seed_products_backfilled.json

Writes: seed_products_final.json
Prints before/after record counts and a list of every subcategory dropped.
"""

import json
import sys

DROP_SUBCATEGORIES = {
    "Kitchen", "Car", "Bath", "Home Décor", "Bedroom",
    "Stationery & Planners", "Porch & Backyard",
    "Heartfelt Gift with Words", "Gift Card", "Marketing", "Dummy",
    "Beach, Camping & Outdoor", "Unique & Fun Treasures",
    "Happy Bags & Totes", "Blankets",
}

TARGET_STORE = "naturallife.com"


def main():
    if len(sys.argv) < 2:
        print("Usage: python filter_naturallife_apparel.py <path_to_seed_products_backfilled.json>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = input_path.replace(".json", "").rsplit("_backfilled", 1)[0] + "_final.json"

    print(f"Loading {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        products = json.load(f)

    print(f"Loaded {len(products)} total products.\n")

    kept = []
    dropped = []

    for p in products:
        if p.get("source_id") == TARGET_STORE and p.get("subcategory") in DROP_SUBCATEGORIES:
            dropped.append(p)
        else:
            kept.append(p)

    print(f"naturallife.com records before filter: "
          f"{sum(1 for p in products if p.get('source_id') == TARGET_STORE)}")
    print(f"naturallife.com records dropped:       {len(dropped)}")
    print(f"naturallife.com records kept:          "
          f"{sum(1 for p in kept if p.get('source_id') == TARGET_STORE)}\n")

    print(f"Total records before: {len(products)}")
    print(f"Total records after:  {len(kept)}\n")

    print("Dropped subcategory breakdown:")
    from collections import Counter
    subcat_counts = Counter(p["subcategory"] for p in dropped)
    for subcat, count in subcat_counts.most_common():
        print(f"  {subcat}: {count}")

    print(f"\nWriting filtered file to {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
