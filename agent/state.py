"""
Shared state schema for Kiran's Digital Closet agent graph.

This maps directly to Section 6 ("What does it need to remember?") of the
Project Master Ledger:
  - Session state (this TypedDict): current upload, extracted attributes,
    retrieved candidates, retry counters.
  - Persistent state (via Mem0, outside this graph): user style preferences,
    disliked colors/brands, saved closet items across sessions. The graph
    reads persistent state through get_user_preferences and writes to it
    through save_to_digital_closet -- it does not own that storage itself.
"""

from typing import Optional, TypedDict


class ClosetAgentState(TypedDict, total=False):
    # --- input ---
    image_path: str
    user_id: str

    # --- vision extraction ---
    attributes: dict            # e.g. {"neck_type": "v-neck", "sleeve_type": "long sleeve", ...}
    vision_confidence: float    # 0.0-1.0, drives the re-upload branch

    # --- query construction / retrieval ---
    query: dict                 # complementary-item query built by the orchestration LLM
    search_results: list        # candidates from search_brand_inventory
    broadened: bool             # guards against infinite broaden loops (broaden once, then stop)

    # --- personalization ---
    user_preferences: dict      # from get_user_preferences (Mem0)

    # --- ranking / output ---
    ranked_recommendations: list
    styling_note: str

    # --- human-in-the-loop / write path ---
    user_wants_to_save: bool    # set externally when the user clicks "save" on a card
    approved_item_id: Optional[str]
    saved: bool

    # --- error handling / retries ---
    retry_counts: dict          # per-tool retry counter, e.g. {"search_brand_inventory": 1}
    status_message: Optional[str]  # plain-language message surfaced to the user on failure/branching

    # --- trace, useful for the demo video / debugging ---
    trace: list

    # --- demo/testing only: forces the search-failure branch to fire once,
    # so demo.py can actually exercise the retry path on demand. Not part
    # of the real runtime state. Vision/build flags force every attempt to
    # fail, proving those nodes degrade gracefully after retries exhaust. ---
    _force_search_failure_once: bool
    _force_zero_results: bool
    _force_vision_failure: bool
    _force_build_query_failure: bool
