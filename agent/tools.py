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
  - remember_user_preference (write, requires human approval -- gated in graph.py,
                               not here)
  - save_to_digital_closet   (write, requires human approval -- gated in graph.py,
                               not here)
  - vision_extract_attributes (not an MCP tool per se, but the vision model call)
  - build_complementary_query (orchestration LLM call)
  - rank_and_style             (orchestration LLM call)
"""

import base64
import json
import mimetypes
import os
import re
from datetime import datetime, timezone

import requests
from PIL import Image
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


class ToolError(Exception):
    """Raised by a tool stub to simulate a transient failure for the retry-once demo."""
    pass


_PINECONE_CLIENT = None
_CLIP_MODEL = None
_MEM0_CLIENT = None

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


class _LocalFakeMem0Store:
    """
    Temporary, free, in-process stand-in for Mem0 -- used automatically
    when MEM0_API_KEY isn't set yet. Mimics only the two calls this
    project actually uses (add/get_all), matching their real shapes
    closely enough that _extract_preferences_from_memories doesn't need
    to know which one it's talking to.
    """
    def __init__(self):
        self._by_user = {}

    def add(self, messages, user_id):
        text = messages[0]["content"]
        self._by_user.setdefault(user_id, []).append(text)
        return {"results": [{"memory": text}]}

    def get_all(self, user_id=None, filters=None):
        if filters:
            user_id = filters.get("user_id")
        return [{"memory": t} for t in self._by_user.get(user_id, [])]


_LOCAL_FAKE_MEM0 = _LocalFakeMem0Store()


def _get_mem0_client():
    """
    Hosted Mem0 Platform client when MEM0_API_KEY is set; local in-memory
    fake otherwise, so graph wiring can be developed without waiting on
    hosted credits. The fake is not persistent across process restarts.
    """
    global _MEM0_CLIENT
    if _MEM0_CLIENT is None:
        api_key = os.environ.get("MEM0_API_KEY")
        if not api_key:
            print("MEM0_API_KEY not set -- using a local in-memory store "
                  "for now (preferences won't persist across process "
                  "restarts). Set MEM0_API_KEY once your Mem0 credits are "
                  "active to switch to the real hosted store automatically.")
            _MEM0_CLIENT = _LOCAL_FAKE_MEM0
        else:
            from mem0 import MemoryClient
            _MEM0_CLIENT = MemoryClient(api_key=api_key)
    return _MEM0_CLIENT


def remember_user_preference(user_id: str, preference_text: str,
                             simulate_failure: bool = False) -> dict:
    """
    New WRITE tool -- saves a stated preference into Mem0 so a later
    get_user_preferences call can retrieve it. Mem0 does its own fact
    extraction from raw text in hosted mode; the local fake stores the
    same raw text for graph-wiring demos.

    graph.py must only call this from the remember_preference node, which
    sits behind the same interrupt_before gate save_to_digital_closet
    uses. This function assumes approval has already happened.
    """
    if simulate_failure:
        raise ToolError("Mem0 preference write failed")
    client = _get_mem0_client()
    return client.add(
        messages=[{"role": "user", "content": preference_text}],
        user_id=user_id,
    )


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
# Vision extraction -- GLM-5.3-Flash via Fireworks (primary) / Nebius
# (fallback). REAL as of step 4 + garment_type correction (applied here
# together, in one merged pass).
# ----------------------------------------------------------------------

# Mirrors backfill_attributes.py's NECK_TYPES / SLEEVE_TYPES / PATTERNS /
# SILHOUETTES exactly, so a live user upload and the catalog backfill agree
# on the same enum values. Keep these two lists in sync by hand until
# they're pulled into one shared constants module (flagged as known
# duplication debt, not fixed here to keep this a single-purpose diff).
NECK_TYPES = [
    "v-neck", "round neck", "boat neck", "halter", "off shoulder",
    "sweetheart", "collared", "turtleneck", "cowl", "square neck",
    "high neck", "scoop neck", "bateau",
]

SLEEVE_TYPES = [
    "sleeveless", "short sleeve", "long sleeve", "3/4 sleeve",
    "cap sleeve", "puff sleeve", "strapless",
]

PATTERNS = [
    "floral", "striped", "solid", "polka dot", "animal print",
    "plaid", "checked", "geometric", "abstract",
]

SILHOUETTES = [
    "a-line", "bodycon", "wrap", "cut out", "fitted", "relaxed",
    "oversized", "structured", "flowy",
]

# primary_color is deliberately NOT a controlled vocab -- color language is
# open-ended ("sage green", "dusty rose") and build_complementary_query
# already treats it as free text. Only empty/garbage values are rejected,
# never off-list ones.

GARMENT_TYPES = ["top", "bottom", "dress", "outerwear", "accessory"]

# Which of the other 4 extracted fields are actually expected to be
# determinable for a given garment_type -- fixes the "pants unfairly
# penalized to 0.6 confidence" problem: neck_type/sleeve_type are dropped
# from the denominator entirely for bottoms and accessories, instead of
# counting as "missing" against a fixed field count.
FIELD_APPLICABILITY = {
    "top":       ["neck_type", "sleeve_type", "pattern", "silhouette", "primary_color"],
    "dress":     ["neck_type", "sleeve_type", "pattern", "silhouette", "primary_color"],
    "outerwear": ["neck_type", "sleeve_type", "pattern", "silhouette", "primary_color"],
    "bottom":    ["pattern", "silhouette", "primary_color"],
    "accessory": ["pattern", "primary_color"],
}

GARMENT_TYPE_ALIASES = {
    "top": "top", "tops": "top",
    "bottom": "bottom", "bottoms": "bottom",
    "dress": "dress", "dresses": "dress",
    "outerwear": "outerwear", "outerwears": "outerwear",
    "accessory": "accessory", "accessories": "accessory",
}

VISION_SYSTEM_PROMPT = (
    "Respond with ONLY the JSON object, no reasoning, no explanation, no "
    "markdown code fences."
)

VISION_PROMPT = """You are tagging a single fashion item photo uploaded by a
user to their digital closet. Look at the image and return ONLY a JSON
object with any of these fields you can confidently determine. Omit a field
entirely if unsure -- do not guess.

{
  "garment_type": one of """ + json.dumps(GARMENT_TYPES) + """ -- choose this first, before the other fields,
  "neck_type": one of """ + json.dumps(NECK_TYPES) + """,
  "sleeve_type": one of """ + json.dumps(SLEEVE_TYPES) + """,
  "pattern": one of """ + json.dumps(PATTERNS) + """,
  "silhouette": one of """ + json.dumps(SILHOUETTES) + """,
  "primary_color": the item's dominant color, as a short free-text phrase (e.g. "black", "sage green")
}
"""

FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
FIREWORKS_MODEL = "accounts/fireworks/models/glm-5p3-flash"
NEBIUS_URL = "https://api.tokenfactory.nebius.com/v1/chat/completions"
NEBIUS_MODEL = "zai-org/GLM-5.3-Flash"


def _encode_image_data_uri(image_path: str) -> str:
    """
    Base64 data-URI encode the user's uploaded photo for the chat-completions
    image_url field. Unlike backfill_attributes.py (which points at a
    catalog product's already-public image_url), this is a local file from
    the upload -- there's no URL to hand the API, so it has to be inlined.
    Raises ToolError (not a bare PIL/OSError) on a missing/corrupt file.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
    except (FileNotFoundError, OSError) as e:
        raise ToolError(f"vision_extract_attributes: couldn't open image_path {image_path!r}: {e}")

    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"  # closet photos are almost always jpg/png; safe default

    with open(image_path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def _extract_json(content: str) -> dict:
    """Same fix as the ledger's GLM-5.3-Flash note: pull the JSON object out
    with a DOTALL regex rather than a naive strip/json.loads, since the
    model can still wrap it in commentary even with the reasoning-
    suppression system prompt."""
    if not content:
        raise ValueError("empty response content")
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in response: {content[:200]!r}")
    return json.loads(match.group(0))


def _match_vocab_tolerant(value, vocab):
    """
    Like the exact-match vocab check used for neck_type/sleeve_type/pattern/
    silhouette below, but tolerant of a singular/plural mismatch specifically
    (e.g. the model saying "tops" when the vocab says "top"). Needed for
    garment_type because target_category elsewhere in this codebase already
    uses the plural form ("bottoms", "outerwear", "accessories"), so a model
    conflating the two spellings is a real, likely failure mode.

    Uses an explicit alias map (GARMENT_TYPE_ALIASES) rather than a generic
    "strip a trailing s" heuristic -- a naive rstrip("s") approach fails on
    "dress"/"dresses" specifically, because "dress" already ends in an "s"
    that isn't a plural marker.
    """
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in vocab:
        return v
    canonical = GARMENT_TYPE_ALIASES.get(v)
    if canonical in vocab:
        return canonical
    return None


def _validate_and_clean(parsed: dict) -> dict:
    """Controlled-vocab drop behavior -- a value outside the enum is
    dropped, not kept as free text, so Pinecone metadata / text_query
    wording stays consistent with the catalog side. primary_color is exempt
    (see note above)."""
    clean = {}

    garment_type = _match_vocab_tolerant(parsed.get("garment_type"), GARMENT_TYPES)
    if garment_type:
        clean["garment_type"] = garment_type

    for field, vocab in [
        ("neck_type", NECK_TYPES), ("sleeve_type", SLEEVE_TYPES),
        ("pattern", PATTERNS), ("silhouette", SILHOUETTES),
    ]:
        val = parsed.get(field)
        val = val.strip().lower() if isinstance(val, str) else None
        if val in vocab:
            clean[field] = val

    color = parsed.get("primary_color")
    if isinstance(color, str) and color.strip():
        clean["primary_color"] = color.strip().lower()

    return clean


def _compute_confidence(clean_attributes: dict) -> float:
    """
    garment_type itself is always expected (every photo has SOME garment
    type) and is weighted the same as one applicable field. The other 4
    fields are only counted against the subset FIELD_APPLICABILITY says is
    expected for that garment_type -- so a bottom-item photo is graded out
    of 4 total (garment_type + 3 applicable fields), not 6.

    If garment_type itself couldn't be determined, there's no way to know
    which fields even apply -- falls back to grading against the full
    5-field set (garment_type absent counts as a miss, same as any other
    field). This is deliberately the *harshest* case, not a lenient
    default: an image the model can't even categorize should score low.
    """
    garment_type = clean_attributes.get("garment_type")
    if garment_type and garment_type in FIELD_APPLICABILITY:
        applicable = FIELD_APPLICABILITY[garment_type]
    else:
        applicable = ["neck_type", "sleeve_type", "pattern", "silhouette", "primary_color"]

    total = len(applicable) + 1  # +1 for garment_type itself
    hit = sum(1 for f in applicable if f in clean_attributes)
    hit += 1 if garment_type else 0
    return round(hit / total, 2)


def _call_glm(url: str, model: str, api_key: str, image_data_uri: str, timeout: int = 30) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "max_tokens": 800,  # >= 600 per ledger; headroom above the observed minimum
        "temperature": 0,
        "messages": [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ],
            },
        ],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    response_json = resp.json()
    choice = response_json["choices"][0]
    content = choice["message"].get("content")
    if not content:
        raise ValueError(f"empty content, finish_reason={choice.get('finish_reason')!r}")
    return content


def vision_extract_attributes(image_path: str, simulate_failure: bool = False) -> dict:
    """
    REAL IMPLEMENTATION (step 4, garment_type-aware). Same signature/return
    shape as the old stub: {"attributes": {...}, "confidence": float 0.0-1.0}.

    Provider order per the ledger's Sept 6, 2026 decision: try Fireworks
    first; on ANY exception (HTTP error, timeout, malformed/empty response,
    JSON-parse failure), retry once against Nebius before giving up. Only
    raises ToolError -- letting graph.py's call_with_retry take one more
    full pass at both providers -- if Nebius also fails (or isn't
    configured).
    """
    if simulate_failure:
        raise ToolError("Vision model request failed")

    image_data_uri = _encode_image_data_uri(image_path)

    fireworks_key = os.environ.get("FIREWORKS_API_KEY")
    nebius_key = os.environ.get("NEBIUS_API_KEY")

    last_err = None
    parsed = None

    if fireworks_key:
        try:
            content = _call_glm(FIREWORKS_URL, FIREWORKS_MODEL, fireworks_key, image_data_uri)
            parsed = _extract_json(content)
        except Exception as e:
            last_err = e
    else:
        last_err = RuntimeError("FIREWORKS_API_KEY not set")

    if parsed is None:
        if nebius_key:
            try:
                content = _call_glm(NEBIUS_URL, NEBIUS_MODEL, nebius_key, image_data_uri)
                parsed = _extract_json(content)
            except Exception as e:
                last_err = e
        elif last_err is None:
            last_err = RuntimeError("NEBIUS_API_KEY not set")

    if parsed is None:
        raise ToolError(f"vision_extract_attributes: both providers failed -- {last_err}")

    attributes = _validate_and_clean(parsed)
    confidence = _compute_confidence(attributes)
    return {"attributes": attributes, "confidence": confidence}


# ----------------------------------------------------------------------
# Orchestration LLM: query construction
# ----------------------------------------------------------------------

# Plural form to match the existing target_category convention already in
# use ("bottoms", "outerwear", "accessories" in the system prompt's own
# examples) -- garment_type itself stays singular (describes the ONE
# uploaded item), target_category stays plural (describes the category
# being searched FOR).
COMPLEMENTARY_CATEGORY_FALLBACK = {
    "top": "bottoms",
    "bottom": "tops",
    "dress": "outerwear",   # judgment call -- revisit once you have real usage
    "outerwear": "tops",    # judgment call -- revisit once you have real usage
    "accessory": "tops",    # judgment call -- revisit once you have real usage
}


def _fallback_target_category(garment_type, broaden: bool) -> str:
    """Replaces the literal `None if broaden else "bottoms"` default."""
    if broaden:
        return None
    return COMPLEMENTARY_CATEGORY_FALLBACK.get(garment_type, "bottoms")


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
    if attributes.get("garment_type"):
        user_prompt += f"\nThis item's garment_type is: {attributes['garment_type']}."
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
            "model": "nvidia/Nemotron-3_5-Lightning",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 400,
            "temperature": 0.3,
        },
        timeout=60,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ToolError(f"build_complementary_query: no JSON found in LLM response: {content!r}")
    query = json.loads(match.group(0))

    garment_type = attributes.get("garment_type")
    query.setdefault("target_category", _fallback_target_category(garment_type, broaden))
    query.setdefault("compatible_colors", [])
    query.setdefault("occasion", attributes.get("occasion", "casual"))
    query.setdefault("text_query", "")
    if broaden:
        query["target_category"] = None
    return query


# Soft keyword match against whatever raw per-brand category string
# Pinecone metadata actually holds -- deliberately NOT a strict equality
# check. This project already learned that lesson once: occasion was
# removed as a hard Pinecone filter for exactly this reason (raw per-brand
# tag strings like "office outfits" won't reliably $eq-match a normalized
# enum). Same risk applies to category, so this is a tolerant substring
# match, not an exact one, and it's permissive on anything it can't
# confidently judge, rather than restrictive.
CATEGORY_KEYWORDS = {
    "tops": ["top", "blouse", "shirt", "tee", "tank", "cami", "sweater",
             "sweatshirt", "hoodie", "bodysuit"],
    "bottoms": ["bottom", "pant", "trouser", "short", "jean", "skirt",
                "skort", "legging"],
    "dresses": ["dress", "gown", "jumpsuit", "romper"],
    "outerwear": ["jacket", "coat", "blazer", "cardigan", "outerwear", "vest"],
    "accessories": ["accessory", "accessories", "bag", "jewelry", "belt",
                     "scarf", "hat"],
}


def _category_matches_target(candidate_category: str, target_category: str,
                             candidate_name: str = "") -> bool:
    """
    Checks candidate_name FIRST, falling back to candidate_category only
    when name gives no signal. This order is deliberate: live testing
    (Sept 13) found real catalog records where category confidently
    states the WRONG macro-category (e.g. a maxi skirt filed under
    category="Tops") -- checking category first let those records slip
    through undetected. name is free text but is directly authored per
    product and reliably contains the actual garment word, making it the
    more trustworthy signal for this catalog.

    For each text source in turn: a match against the TARGET category's
    own keywords is a confident keep; a match against a DIFFERENT
    category's keywords is a confident exclude; no match at all moves on
    to the next text source. If neither gives any signal, permissive
    default still applies -- don't punish incomplete data, don't guess.
    """
    if not target_category:
        return True
    target_category = target_category.lower()
    keywords = CATEGORY_KEYWORDS.get(target_category)
    if not keywords:
        return True

    for text in (candidate_name, candidate_category):
        if not text:
            continue
        low = text.lower()
        if any(kw in low for kw in keywords):
            return True
        for other_cat, other_kws in CATEGORY_KEYWORDS.items():
            if other_cat == target_category:
                continue
            if any(kw in low for kw in other_kws):
                return False

    return True


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

    # Enforce target_category using metadata already being returned per
    # candidate, instead of relying on text_query wording alone -- live
    # testing (Sept 13) showed wording alone isn't reliable enough on its
    # own. Skipped entirely when target_category is unset (the
    # broaden-query path), which already exists as this filter's safety
    # valve if it's ever too strict.
    target_category = query.get("target_category")
    if target_category:
        candidates = [
            c for c in candidates
            if _category_matches_target(c["category"], target_category, c["name"])
        ]

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

# Keeps the exact stub schema rank_and_style already depends on
# (preferences.get("budget_max", 9999)) so its caller never has to change.
# A fresh user with nothing in Mem0 yet gets these defaults -- unfiltered
# search, not a broken/missing budget_max.
DEFAULT_PREFERENCES = {
    "disliked_colors": [],
    "preferred_brands": [],
    "budget_max": 9999.0,
}

# Heuristic keyword scan, not an enum -- primary_color is deliberately
# free text project-wide, so this is intentionally a scan list for
# catching common cases in a stated preference, not a validator.
COMMON_COLOR_WORDS = [
    "black", "white", "red", "blue", "green", "yellow", "orange", "purple",
    "pink", "brown", "grey", "gray", "beige", "navy", "cream", "tan",
    "maroon", "olive", "teal", "gold", "silver", "ivory", "khaki",
    "burgundy", "lavender", "mint", "coral", "mustard", "rust",
]

NEGATION_PHRASES = [
    "don't like", "dont like", "doesn't like", "dislike", "hate",
    "do not like", "does not like", "avoid", "not a fan of", "no more",
    "never wants", "doesn't want", "does not want",
]

# Maps a human-readable brand mention to the exact domain string
# search_brand_inventory's catalog already uses.
BRAND_NAME_TO_DOMAIN = {
    "saboskirt": "saboskirt.com",
    "petal and pup": "petalandpup.com",
    "petalandpup": "petalandpup.com",
    "meshki": "meshki.us",
    "bohme": "bohme.com",
    "natural life": "naturallife.com",
    "naturallife": "naturallife.com",
    "oh polly": "ohpolly.com",
    "ohpolly": "ohpolly.com",
    "red dress": "reddress.com",
    "reddress": "reddress.com",
}

_BUDGET_TRIGGER_WORDS = ("budget", "under $", "under$", " max", "at most", "no more than")
_BUDGET_NUMBER_PATTERN = re.compile(r"\$?\s?(\d{2,4}(?:\.\d{1,2})?)")


def _memory_text(memory: dict) -> str:
    """Mem0 may return the fact under 'memory' or 'text' by SDK version."""
    return memory.get("memory") or memory.get("text") or ""


def _get_all_memories(client, user_id: str) -> list:
    """Normalize hosted Mem0 and local fake get_all response shapes."""
    try:
        memories = client.get_all(filters={"user_id": user_id}) or []
    except (TypeError, ValueError):
        memories = client.get_all(user_id=user_id) or []
    if isinstance(memories, dict):
        return memories.get("results", [])
    return memories


def _extract_preferences_from_memories(memory_texts: list) -> dict:
    """
    Translation layer: Mem0 stores/returns free-text facts, but
    rank_and_style reads a fixed {disliked_colors, preferred_brands,
    budget_max} shape. Keyword scan, not an LLM call.
    """
    disliked_colors = set()
    preferred_brands = set()
    budget_candidates = []

    for text in memory_texts:
        low = text.lower()

        is_negative = any(neg in low for neg in NEGATION_PHRASES)
        if is_negative:
            for color in COMMON_COLOR_WORDS:
                if color in low:
                    disliked_colors.add(color)

        for brand_name, domain in BRAND_NAME_TO_DOMAIN.items():
            if brand_name in low:
                preferred_brands.add(domain)

        if any(trigger in low for trigger in _BUDGET_TRIGGER_WORDS):
            match = _BUDGET_NUMBER_PATTERN.search(low)
            if match:
                budget_candidates.append(float(match.group(1)))

    return {
        "disliked_colors": sorted(disliked_colors) if disliked_colors
        else DEFAULT_PREFERENCES["disliked_colors"],
        "preferred_brands": sorted(preferred_brands) if preferred_brands
        else DEFAULT_PREFERENCES["preferred_brands"],
        "budget_max": min(budget_candidates) if budget_candidates
        else DEFAULT_PREFERENCES["budget_max"],
    }


def get_user_preferences(user_id: str) -> dict:
    """
    Same return shape as the old stub, now backed by Mem0 or the local
    fake. A fresh user with zero stated preferences gets defaults.
    """
    client = _get_mem0_client()
    memories = _get_all_memories(client, user_id)
    memory_texts = [_memory_text(m) for m in memories]
    return _extract_preferences_from_memories(memory_texts)


# ----------------------------------------------------------------------
# Orchestration LLM: ranking + styling copy
# ----------------------------------------------------------------------

def rank_and_style(candidates: list, preferences: dict,
                   attributes: dict = None) -> tuple[list, str]:
    """
    Filters out over-budget candidates, deprioritizes (never drops)
    disliked colors, and lightly boosts preferred brands -- then asks
    Llama-3.3-70B (Nebius) to write a short styling note grounded in the
    actual owned-item attributes and top-ranked candidates. Never raises:
    styling-copy failures fall back to deterministic copy from real names.
    """
    attributes = attributes or {}
    budget_max = preferences.get("budget_max", 9999)
    disliked_colors = [c.lower() for c in preferences.get("disliked_colors", [])]
    preferred_brands = [b.lower() for b in preferences.get("preferred_brands", [])]

    affordable = [c for c in candidates if c.get("price", 0) <= budget_max]

    def _rank_key(candidate):
        name_low = (candidate.get("name") or "").lower()
        penalty = 0.15 if any(dc in name_low for dc in disliked_colors) else 0.0
        bonus = 0.05 if (candidate.get("brand") or "").lower() in preferred_brands else 0.0
        return candidate.get("score", 0.0) - penalty + bonus

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
                "model": "nvidia/Nemotron-3_5-Lightning",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 150,
                "temperature": 0.5,
            },
            timeout=20,
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


# ----------------------------------------------------------------------
# save_to_digital_closet (WRITE tool -- human approval enforced in graph.py,
# never here)
# ----------------------------------------------------------------------

_CLOSET_DIR = os.path.join(os.path.dirname(__file__), "data", "closets")


def _closet_file_path(user_id: str) -> str:
    safe_user_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", user_id).strip("._")
    if not safe_user_id:
        raise ToolError("user_id is required to save closet items")
    return os.path.join(_CLOSET_DIR, f"{safe_user_id}.json")


def get_saved_closet_items(user_id: str) -> list:
    path = _closet_file_path(user_id)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("items", [])


def save_to_digital_closet(user_id: str, item: dict) -> bool:
    """
    Persist the approved recommendation into a tiny local JSON closet store.

    This function assumes approval has ALREADY happened -- graph.py must
    never call this without having passed through the interrupt_before gate.
    """
    if not isinstance(item, dict):
        raise ToolError("save_to_digital_closet expected a full item dict")
    item_id = item.get("id")
    if not item_id:
        raise ToolError("save_to_digital_closet item is missing id")

    os.makedirs(_CLOSET_DIR, exist_ok=True)
    path = _closet_file_path(user_id)
    items = get_saved_closet_items(user_id)

    saved_item = dict(item)
    saved_item["saved_at"] = datetime.now(timezone.utc).isoformat()

    replaced = False
    for i, existing in enumerate(items):
        if existing.get("id") == item_id:
            items[i] = saved_item
            replaced = True
            break
    if not replaced:
        items.append(saved_item)

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)
    return True
