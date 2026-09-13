# Scenario 2 Category-Leak Root Cause + Dynamic Styling Note

New file. Does not edit any existing doc, `tools.py`, or `graph.py` directly -- per the standing "new files only" rule. Answers both things flagged in the live Scenario 2 output: (1) why bottoms items still leaked through the Step 6 filter even though `_category_matches_target` itself is correct and unit-verified, and (2) why the styling note is identical across every scenario.

## Part 1 -- Why Scenario 2 Still Leaked Bottoms Items

Your `_category_matches_target` logic is NOT broken -- its own unit tests (9/9) and my independent re-check both pass. The bug is upstream of it: the filter only ever looks at `c["category"]`, and this catalog's `category` metadata is already documented as unreliable in your own project (`claude/step3_handoff_note.md`): a real pair of pants was found filed under `category="Dresses"` / `subcategory="Bottoms"` -- the 7 brands' own taxonomy is inconsistent, which is exactly why `category` was deliberately never used as a hard Pinecone filter in the first place.

`_category_matches_target`'s permissive design says "if `candidate_category` is missing/empty, keep it -- don't punish incomplete metadata." That's the right instinct for data the filter can't judge, but it means: **any candidate whose `category` field is empty sails through untouched, regardless of what it actually is.** Looking at the real leaked items -- `Loretta Denim Maxi Skirt`, `Winslet Faux Leather Maxi Skirt`, `Amirah Midi Skirt`, `Faith Suiting Wide Leg Pant`, `Bobby Oversized Jean` -- every one of them has the garment type spelled out in its own **name**, which `_to_candidate` already captures (`"name": metadata.get("name", "")`) and which is essentially always populated (unlike `category`). The fix does not need new keywords -- `skirt`, `pant`, and `jean` are already in `CATEGORY_KEYWORDS["bottoms"]` -- it needs a second text source to check when `category` does not help.

### The Fix -- `_category_matches_target` Also Checks `name`, With Real Exclusion Logic

```python
def _category_matches_target(candidate_category: str, target_category: str,
                              candidate_name: str = "") -> bool:
    """
    Checks candidate_category first, falling back to candidate_name when
    category metadata is missing/unhelpful -- name is free text but almost
    always present and reliably contains the garment word itself ("Skirt",
    "Pants", "Blouse"), unlike this catalog's known-inconsistent category
    field (see step3_handoff_note.md: a pair of pants filed under
    category="Dresses").

    For each text source in turn: a match against the TARGET category's own
    keywords is a confident keep; a match against a DIFFERENT category's
    keywords is a confident exclude (this is new -- the original version
    only ever said "keep," never "exclude based on text content"); no match
    at all moves on to the next text source. If neither category nor name
    gives any signal, the original permissive default still applies --
    don't punish incomplete data, don't guess.
    """
    if not target_category:
        return True
    keywords = CATEGORY_KEYWORDS.get(target_category.lower())
    if not keywords:
        return True

    for text in (candidate_category, candidate_name):
        if not text:
            continue
        low = text.lower()
        if any(kw in low for kw in keywords):
            return True
        for other_cat, other_kws in CATEGORY_KEYWORDS.items():
            if other_cat == target_category.lower():
                continue
            if any(kw in low for kw in other_kws):
                return False

    return True
```

And the one-line call-site change in `search_brand_inventory` (same spot Step 6 already touched):

```python
target_category = query.get("target_category")
if target_category:
    candidates = [
        c for c in candidates
        if _category_matches_target(c["category"], target_category, c["name"])
    ]
```

### Verified

Ran for real in this session, not just reasoned through.

Reproduced the actual Scenario 2 output as test input:

```text
Previously-leaked bottoms items (should now be EXCLUDED):
  [PASS] 'Loretta Denim Maxi Skirt - Ecru' -> False
  [PASS] 'Winslet Faux Leather Maxi Skirt - Bone' -> False
  [PASS] 'Amirah Midi Skirt - Lemon' -> False
  [PASS] 'Faith Suiting Wide Leg Pant - Cacao Brown' -> False
  [PASS] 'Bobby Oversized Jean With Front Pleat - Summer Blue' -> False

Correctly-kept tops items (should remain KEPT):
  [PASS] 'Lace Layering Top - Black' -> True
  [PASS] 'One Size Cotton Long Sleeve Easy V-Neck Tee - Charcoal Floral' -> True
  [PASS] 'Prettiest Cotton Lace Blouse - White' -> True
  [PASS] 'Hang Around Cotton Tunic - Dark Navy' -> True   (kept via final
         permissive fallback -- "tunic" isn't in any keyword list, so this
         is a "can't judge, don't punish" keep, same as before, not a bug)
  [PASS] 'Lily Cotton Long Sleeve Tee Shirt - Woodcut Floral Bronze' -> True

Original Step 6 unit tests re-run against the new signature -- still pass:
  [PASS] category='Skirts' target='tops' -> False
  [PASS] category='Blouses' target='tops' -> True
  [PASS] category='Pants' target='bottoms' -> True
  [PASS] category='Maxi Dresses' target='bottoms' -> False
  [PASS] category='' target='tops' -> True
  [PASS] target_category=None -> True (broadened-query case, untouched)
  [PASS] unrecognized target_category -> True (untouched)

ALL 12 CHECKS PASSED
```

This is pure Python logic re-run against the literal real Scenario 2 recommendation names -- not a hypothetical.

## Part 2 -- Dynamic Styling Note

The identical note across every scenario ("These pair well with a fitted black sleeveless top...") is because `rank_and_style` is still the original stub -- this was always flagged as deliberately deferred in the ledger ("rank_and_style stays a placeholder... recommendation quality, not the agentic bar"), not something Step 4/6 touched. Since you are asking for it now, here is a real implementation using the same Nebius/Llama-3.3-70B call your real `build_complementary_query` already makes -- same endpoint, same model, same JSON-extraction pattern where relevant.

```python
def rank_and_style(candidates: list, preferences: dict, attributes: dict = None) -> tuple:
    """
    Filters out over-budget candidates, deprioritizes (never drops)
    disliked colors, and lightly boosts preferred brands -- then asks
    Llama-3.3-70B (Nebius) to write a short styling note grounded in the
    ACTUAL owned-item attributes and the ACTUAL top-ranked candidates,
    instead of a fixed placeholder string. Never raises -- per the
    ledger's own design (get_user_preferences/rank_and_style degrade
    inline rather than propagate failures through call_with_retry), a
    styling-copy failure falls back to a deterministic templated note
    built from the real candidate names, so a slow/broken LLM call never
    turns a successful search into a visible failure.
    """
    attributes = attributes or {}
    budget_max = preferences.get("budget_max", 9999)
    disliked_colors = [c.lower() for c in preferences.get("disliked_colors", [])]
    preferred_brands = [b.lower() for b in preferences.get("preferred_brands", [])]

    affordable = [c for c in candidates if c.get("price", 0) <= budget_max]

    def _rank_key(c):
        name_low = (c.get("name") or "").lower()
        penalty = 0.15 if any(dc in name_low for dc in disliked_colors) else 0.0
        bonus = 0.05 if (c.get("brand") or "").lower() in preferred_brands else 0.0
        return c.get("score", 0.0) - penalty + bonus

    ranked = sorted(affordable, key=_rank_key, reverse=True)
    note = _generate_styling_note(ranked[:3], attributes)
    return ranked, note


def _generate_styling_note(top_candidates: list, attributes: dict) -> str:
    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key or not top_candidates:
        return _fallback_styling_note(top_candidates, attributes)

    system_prompt = (
        "You are a fashion stylist assistant. Given the attributes of an "
        "item a user already owns and a short list of complementary "
        "products actually being recommended to pair with it, write ONE "
        "short (1-2 sentence) styling note grounded ONLY in the items "
        "given -- name the actual pieces by name, not a generic template. "
        "No JSON, no markdown, plain text only."
    )
    user_prompt = (
        f"Owned item attributes: {json.dumps(attributes)}\n"
        f"Top recommended pieces: "
        f"{json.dumps([{'name': c.get('name', ''), 'brand': c.get('brand', '')} for c in top_candidates])}"
    )
    try:
        resp = requests.post(
            "https://api.studio.nebius.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "meta-llama/Llama-3.3-70B-Instruct",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 150,
                "temperature": 0.5,
            },
            timeout=15,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        return content or _fallback_styling_note(top_candidates, attributes)
    except Exception:
        return _fallback_styling_note(top_candidates, attributes)


def _fallback_styling_note(top_candidates: list, attributes: dict) -> str:
    names = [c.get("name", "").strip() for c in top_candidates if c.get("name")]
    if not names:
        return "Here are a few pieces that could work well with this item."
    if len(names) > 2:
        pieces = ", ".join(names[:-1]) + f", and {names[-1]}"
    elif len(names) == 2:
        pieces = f"{names[0]} and {names[1]}"
    else:
        pieces = names[0]
    color = attributes.get("primary_color", "")
    color_phrase = f"your {color} item" if color else "this item"
    return f"{pieces} could pair well with {color_phrase}."
```

Requires one call-site change in `graph.py`'s `node_rank_and_style` to pass `attributes` through (currently only passes `search_results` and `user_preferences`):

```python
def node_rank_and_style(state: ClosetAgentState) -> ClosetAgentState:
    result, err = call_with_retry(
        rank_and_style, state["search_results"], state["user_preferences"],
        state.get("attributes", {}),
    )
```

### Verified

Ran for real, no network call needed for the fallback path.

```text
Track Record Light Heather Blue Jogger Sweatpants, Abrand 94 Wide Jeans - Paloma, and Set For The Day Camel Cotton Scalloped Wide Leg Pants could pair well with your light lavender item.
Lace Layering Top - Black and Prettiest Cotton Lace Blouse - White could pair well with your olive green and cream item.
Solo Top could pair well with your black item.
Here are a few pieces that could work well with this item.
```

Confirms correct grammar for 1, 2, and 3+ item cases and the zero-candidate edge case, using real product names as input. The LLM path itself (`_generate_styling_note`'s live Nebius call) is not verifiable in this environment -- no network access to Nebius here -- so the first live run is the actual test of whether the model's prose reads naturally; the fallback is what guarantees it degrades to something reasonable either way.

## Find/Replace Instructions For Codex

### `agent/tools.py`, Change 1 -- Replace `_category_matches_target`

```text
FIND:
def _category_matches_target(candidate_category: str, target_category: str) -> bool:
    """
    Permissive by default -- only excludes a candidate when reasonably
    confident it's the WRONG macro-category, never when just unsure.
    Three cases nothing gets filtered:
      - no target_category at all (broadened query -- graph.py's existing
        MIN_RESULTS/broaden path is the safety valve for this filter being
        too strict, not a new mechanism this file has to invent)
      - target_category isn't one we have a keyword list for (an
        LLM-invented category word we didn't anticipate -- don't guess)
      - the candidate's own category metadata is missing/empty (same
        graceful-fallback philosophy already used for has_image_vector --
        don't punish incomplete metadata)
    """
    if not target_category:
        return True
    keywords = CATEGORY_KEYWORDS.get(target_category.lower())
    if not keywords:
        return True
    if not candidate_category:
        return True
    low = candidate_category.lower()
    return any(kw in low for kw in keywords)

REPLACE WITH:
def _category_matches_target(candidate_category: str, target_category: str,
                              candidate_name: str = "") -> bool:
    """
    Checks candidate_category first, falling back to candidate_name when
    category metadata is missing/unhelpful -- this catalog's category
    field is known-inconsistent (a pair of pants was found filed under
    category="Dresses"), while name is free text but almost always
    present and reliably contains the garment word itself.

    For each text source in turn: a match against the TARGET category's
    own keywords is a confident keep; a match against a DIFFERENT
    category's keywords is a confident exclude; no match at all moves on
    to the next text source. If neither gives any signal, permissive
    default still applies -- don't punish incomplete data, don't guess.
    """
    if not target_category:
        return True
    keywords = CATEGORY_KEYWORDS.get(target_category.lower())
    if not keywords:
        return True

    for text in (candidate_category, candidate_name):
        if not text:
            continue
        low = text.lower()
        if any(kw in low for kw in keywords):
            return True
        for other_cat, other_kws in CATEGORY_KEYWORDS.items():
            if other_cat == target_category.lower():
                continue
            if any(kw in low for kw in other_kws):
                return False

    return True
```

### `agent/tools.py`, Change 2 -- Update The Call Site In `search_brand_inventory`

```text
FIND:
    target_category = query.get("target_category")
    if target_category:
        candidates = [c for c in candidates if _category_matches_target(c["category"], target_category)]

REPLACE WITH:
    target_category = query.get("target_category")
    if target_category:
        candidates = [
            c for c in candidates
            if _category_matches_target(c["category"], target_category, c["name"])
        ]
```

### `agent/tools.py`, Change 3 -- Replace The `rank_and_style` Stub And Add Its Two New Helpers

Find the current stub:

```text
FIND:
def rank_and_style(candidates: list, preferences: dict) -> tuple:
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

REPLACE WITH:
the three functions (`rank_and_style`, `_generate_styling_note`, `_fallback_styling_note`) given in full above.
```

Confirm `json`, `os`, and `requests` are already imported at the top of `tools.py` before applying. They are expected to be present because `build_complementary_query` already uses all three, so this should not need its own import line.

### `agent/graph.py` -- Update `node_rank_and_style`'s Call Site

```text
FIND:
def node_rank_and_style(state: ClosetAgentState) -> ClosetAgentState:
    result, err = call_with_retry(rank_and_style, state["search_results"], state["user_preferences"])

REPLACE WITH:
def node_rank_and_style(state: ClosetAgentState) -> ClosetAgentState:
    result, err = call_with_retry(
        rank_and_style, state["search_results"], state["user_preferences"],
        state.get("attributes", {}),
    )
```

Stage and commit only these two files (`agent/tools.py`, `agent/graph.py`) for this change, same discipline as the `graph.py` message fix -- leave the pre-existing unrelated modifications alone.

Suggested commit message:

```text
Fix category filter to also check product name; make styling note dynamic (Llama-3.3-70B, grounded fallback)
```

## What To Check On The Next Live Run

- **Scenario 2**: `Recommendations` should now be ALL tops (or at minimum, no more skirts/pants/jeans by name) -- the actual pass/fail bar.
- **Scenario 1 / 2 / 4**: the `Styling note` line should now differ between scenarios and actually name the real top-ranked pieces for that specific run, instead of the identical fixed sentence appearing in all three.
- If the Nebius call in `_generate_styling_note` fails or times out, the note will read like the fallback format ("X, Y, and Z could pair well with your <color> item") rather than crashing or reverting to the old hardcoded sentence -- that is expected, not a bug, if it happens.
