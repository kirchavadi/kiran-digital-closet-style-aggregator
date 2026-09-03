"""
Shopify /products.json scraper + normalizer
=============================================
Pulls product data from the public, unauthenticated /products.json endpoint
that most Shopify stores expose by default, and normalizes it into a common
schema ready for embedding + Pinecone ingestion.

Usage:
    python shopify_scraper.py

Output:
    seed_products.json  -- normalized product list
    scrape_log.json     -- per-store stats (items found, errors, pages hit)
"""

import json
import re
import time
import html
from urllib.parse import urljoin
from dataclasses import dataclass, asdict, field

import requests

# ----------------------------------------------------------------------
# 1. CONFIG — curate this list. Only include domains you've confirmed
#    return JSON (see validate_domains() below before adding here).
# ----------------------------------------------------------------------

STORE_DOMAINS = [
    "saboskirt.com",
    "petalandpup.com",
    "meshki.us",
    "bohme.com",
    "naturallife.com",
    "ohpolly.com",
    "reddress.com",
    # Add more validated Shopify-store domains here.
]

REQUEST_DELAY_SECONDS = 1.5   # be polite -- don't hammer stores
PAGE_LIMIT = 250              # Shopify's max per-page limit
MAX_PAGES_PER_STORE = 40      # safety cap (40 * 250 = 10,000 products/store)
TIMEOUT_SECONDS = 15
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; StylistAggregatorBot/1.0; "
                  "+https://example.com/about-this-bot)"
}


# ----------------------------------------------------------------------
# 2. NORMALIZED SCHEMA
# ----------------------------------------------------------------------

@dataclass
class NormalizedProduct:
    id: str                    # composite: domain + shopify product id
    source_id: str              # domain, used as the "source"
    external_id: str            # shopify's numeric product id
    name: str
    brand: str
    category: str                # coarse bucket, mapped from product_type/tags
    subcategory: str | None
    price: float | None
    original_price: float | None
    currency: str
    sizes: str                  # comma-joined, matches your existing schema
    colors: str                 # JSON-stringified list, matches existing schema
    fabric: str | None
    image_url: str | None
    product_url: str
    description: str
    tags: str
    in_stock: int
    source_name: str


# Very light category mapping -- expand this as you see more product_type
# values come through. Anything unmatched falls back to "Other" and the
# raw product_type/tags are preserved in `subcategory`/`tags` so nothing
# is lost, just uncategorized.
CATEGORY_KEYWORDS = {
    "Dresses": ["dress", "gown", "maxi", "midi dress"],
    "Tops": ["top", "blouse", "shirt", "tee", "tank", "bodysuit", "cami"],
    "Pants": ["pant", "jean", "trouser", "legging", "short"],
    "Skirts": ["skirt"],
    "Outerwear": ["jacket", "coat", "blazer", "cardigan", "vest"],
    "Sweaters": ["sweater", "knit"],
    "Swimwear": ["swim", "bikini", "one-piece"],
    "Jumpsuits": ["jumpsuit", "romper", "overall"],
    "Activewear": ["active", "yoga", "sports bra", "gym"],
    "Loungewear": ["lounge", "pajama", "sleep", "robe"],
}


def map_category(product_type: str, tags: list) -> tuple[str, str | None]:
    haystack = " ".join([product_type or ""] + (tags or [])).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in haystack for kw in keywords):
            return category, (product_type or None)
    return "Other", (product_type or None)


def strip_html(raw_html: str) -> str:
    if not raw_html:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


# ----------------------------------------------------------------------
# 3. FETCH + PAGINATE
# ----------------------------------------------------------------------

def fetch_store_products(domain: str) -> list:
    """Fetch all products from one store's /products.json, paginated."""
    all_raw_products = []
    page = 1

    while page <= MAX_PAGES_PER_STORE:
        url = f"https://{domain}/products.json?limit={PAGE_LIMIT}&page={page}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS)
        except requests.RequestException as e:
            print(f"  [ERROR] {domain} page {page}: {e}")
            break

        if resp.status_code != 200:
            print(f"  [SKIP] {domain} page {page}: HTTP {resp.status_code}")
            break

        try:
            data = resp.json()
        except ValueError:
            print(f"  [SKIP] {domain} page {page}: non-JSON response "
                  f"(endpoint likely disabled)")
            break

        products = data.get("products", [])
        if not products:
            break  # reached the end

        all_raw_products.extend(products)
        print(f"  [OK] {domain} page {page}: {len(products)} products "
              f"(running total: {len(all_raw_products)})")
        page += 1
        time.sleep(REQUEST_DELAY_SECONDS)

    return all_raw_products


# ----------------------------------------------------------------------
# 4. NORMALIZE
# ----------------------------------------------------------------------

def normalize_product(raw: dict, domain: str) -> NormalizedProduct | None:
    variants = raw.get("variants", [])
    images = raw.get("images", [])

    if not variants:
        return None  # skip products with no purchasable variant

    first_variant = variants[0]
    price = _safe_float(first_variant.get("price"))
    compare_at = _safe_float(first_variant.get("compare_at_price"))

    sizes = sorted({
        v.get("option1") for v in variants
        if v.get("option1") and v.get("option1").lower() != "default title"
    })
    colors = sorted({
        v.get("option2") for v in variants
        if v.get("option2")
    })

    category, subcategory = map_category(
        raw.get("product_type", ""), raw.get("tags", [])
    )

    handle = raw.get("handle", "")
    product_url = f"https://{domain}/products/{handle}"
    image_url = images[0].get("src") if images else None

    return NormalizedProduct(
        id=f"{domain}:{raw.get('id')}",
        source_id=domain,
        external_id=str(raw.get("id")),
        name=raw.get("title", "").strip(),
        brand=raw.get("vendor", "").strip() or domain,
        category=category,
        subcategory=subcategory,
        price=price,
        original_price=compare_at if compare_at and compare_at != price else None,
        currency="USD",
        sizes=", ".join(sizes),
        colors=json.dumps(colors),
        fabric=None,  # Shopify doesn't standardize this -- leave for Vision LLM tagging
        image_url=image_url,
        product_url=product_url,
        description=strip_html(raw.get("body_html", "")),
        tags=", ".join(raw.get("tags", [])) if isinstance(raw.get("tags"), list) else str(raw.get("tags", "")),
        in_stock=1 if any(v.get("available") for v in variants) else 0,
        source_name=domain,
    )


def _safe_float(value):
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None


# ----------------------------------------------------------------------
# 5. VALIDATE DOMAINS (run this BEFORE adding a domain to STORE_DOMAINS)
# ----------------------------------------------------------------------

def validate_domain(domain: str) -> bool:
    """Quick check: does this domain expose a working /products.json?"""
    url = f"https://{domain}/products.json?limit=1"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS)
        if resp.status_code != 200:
            return False
        data = resp.json()
        return "products" in data
    except (requests.RequestException, ValueError):
        return False


def validate_domains(candidate_domains: list) -> list:
    valid = []
    for domain in candidate_domains:
        ok = validate_domain(domain)
        print(f"  {'VALID' if ok else 'INVALID'}: {domain}")
        if ok:
            valid.append(domain)
        time.sleep(1)
    return valid


# ----------------------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------------------

def main():
    all_normalized = []
    log = {}

    print(f"Scraping {len(STORE_DOMAINS)} stores...\n")

    for domain in STORE_DOMAINS:
        print(f"-> {domain}")
        raw_products = fetch_store_products(domain)

        normalized = []
        for raw in raw_products:
            product = normalize_product(raw, domain)
            if product:
                normalized.append(product)

        all_normalized.extend(normalized)
        log[domain] = {
            "raw_count": len(raw_products),
            "normalized_count": len(normalized),
        }
        print(f"   -> {len(normalized)} normalized products from {domain}\n")

    # Write outputs
    with open("seed_products.json", "w") as f:
        json.dump([asdict(p) for p in all_normalized], f, indent=2)

    with open("scrape_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nDone. {len(all_normalized)} total products written to "
          f"seed_products.json")


if __name__ == "__main__":
    # Uncomment to validate a new batch of candidate domains before
    # adding them to STORE_DOMAINS above:
    #
    # candidates = ["somenewbrand.com", "anotherbrand.co"]
    # print("Validating candidate domains...")
    # valid = validate_domains(candidates)
    # print(f"\nValid domains: {valid}")

    main()
