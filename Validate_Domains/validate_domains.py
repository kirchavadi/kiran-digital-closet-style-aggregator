"""
Validate candidate Shopify store domains
==========================================
Run this BEFORE shopify_scraper.py. It checks which candidate domains
actually expose a working /products.json endpoint, so you only add
confirmed-working stores to STORE_DOMAINS in the main scraper.

Usage:
    python validate_domains.py

Requires shopify_scraper.py to be in the same folder (imports from it).
"""

from shopify_scraper import validate_domains

# ----------------------------------------------------------------------
# Add any new candidate domains you want to check here. These do NOT
# need to be added to shopify_scraper.py's STORE_DOMAINS yet -- that's
# the whole point of validating first.
# ----------------------------------------------------------------------

CANDIDATE_DOMAINS = [
    "saboskirt.com",
    "petalandpup.com",
    "meshki.us",
    "bohme.com",
    "naturallife.com",
    "ohpolly.com",
    "reddress.com",
    # Add new candidates below, e.g.:
    # "somenewbrand.com",
    # "anotherbrand.co",
]


def main():
    print(f"Validating {len(CANDIDATE_DOMAINS)} candidate domains...\n")

    valid = validate_domains(CANDIDATE_DOMAINS)
    invalid = [d for d in CANDIDATE_DOMAINS if d not in valid]

    print("\n" + "=" * 50)
    print(f"VALID ({len(valid)}) -- safe to add to STORE_DOMAINS:")
    for d in valid:
        print(f"  {d}")

    if invalid:
        print(f"\nINVALID ({len(invalid)}) -- endpoint blocked/disabled:")
        for d in invalid:
            print(f"  {d}")

    print("\nCopy the VALID list above into STORE_DOMAINS in "
          "shopify_scraper.py before running the scraper.")


if __name__ == "__main__":
    main()
