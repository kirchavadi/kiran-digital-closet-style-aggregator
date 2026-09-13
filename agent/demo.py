"""
Demo runner for the Digital Closet agent graph -- step 4 applied.

Runs seven scenarios so every documented branch actually executes:
  1. Happy path: top-item photo, real vision call, present + save.
  2. Bottoms-item photo: exercises the new garment_type fix end to end
     (confidence not penalized for absent neck/sleeve, target_category
     correctly comes back "tops" instead of the old hardcoded "bottoms").
  3. Low-confidence vision tag -> re-upload branch.
  4. Forced search-tool failure -> retry-once -> plain-language message.
  5. Zero search results -> explicit message, not a blank card view.
  6. Forced vision_extract total failure -> graceful stop, no crash.
  7. Forced build_query total failure -> graceful stop, no crash.

IMPORTANT: every scenario below now makes a REAL, BILLED API call to
Fireworks/Nebius (vision_extract_attributes is no longer a stub) and, from
search_brand_inventory onward, to Pinecone. Each scenario invokes the graph
ONCE per real photo -- there is no retry-until-a-particular-confidence
loop. The old version of this file had exactly that kind of loop in the
happy-path and low-confidence scenarios, written for a random stub that no
longer exists; against the real, deterministic vision call that loop would
either do nothing or run forever making live API calls. See
claude/step4_apply_diff_summary.md for the full explanation.

Usage:
    python demo.py
"""

import time

from graph import build_graph
from tools import get_user_preferences


def print_trace(state):
    print("\n--- trace ---")
    for line in state.get("trace", []):
        print(f"  {line}")
    print("-------------\n")


def scenario_happy_path(app):
    print("=" * 60)
    print("SCENARIO 1: happy path (top item photo -> confident tag, save approved)")
    print("=" * 60)

    config = {"configurable": {"thread_id": "demo-happy-1"}}
    initial = {
        "image_path": "test_images/closet_top.jpg",
        "user_id": "kiran-demo-user",
    }

    result = app.invoke(initial, config)

    print(f"Vision confidence: {result.get('vision_confidence')}")
    print(f"Extracted attributes: {result.get('attributes')}")
    print(f"Status message: {result.get('status_message')}")

    if result.get("vision_confidence", 0) < 0.65:
        print("NOTE: this photo scored below the 0.65 threshold for real -- "
              "the graph correctly routed to the re-upload branch instead of "
              "search. Not a bug; if you expected this photo to read "
              "confidently, that's worth a look before recording the demo "
              "video with it.")
        print_trace(result)
        return

    if result.get("ranked_recommendations"):
        print(f"Recommendations: {[r['name'] for r in result['ranked_recommendations']]}")
        print(f"Styling note: {result['styling_note']}")

        print("\n[human-in-the-loop] User clicks 'Save' on the first recommendation...")
        app.update_state(config, {
            "user_wants_to_save": True,
            "approved_item_id": result["ranked_recommendations"][0]["id"],
        })
        final = app.invoke(None, config)  # resume from the interrupt
        print(f"Saved: {final.get('saved')}")
        print_trace(final)
    else:
        print("No recommendations to save -- skipping the save step for this run.")
        print_trace(result)


def scenario_bottoms_photo(app):
    """
    Step 4's garment_type fix, exercised end to end for the first time.
    Requires test_images/closet_bottoms.jpg.
    """
    print("=" * 60)
    print("SCENARIO 2: bottoms item photo -> garment_type fix check")
    print("=" * 60)

    config = {"configurable": {"thread_id": "demo-bottoms-1"}}
    initial = {
        "image_path": "test_images/closet_bottoms.jpg",
        "user_id": "kiran-demo-user",
    }
    result = app.invoke(initial, config)

    attrs = result.get("attributes", {})
    confidence = result.get("vision_confidence")
    garment_type = attrs.get("garment_type")

    print(f"Vision confidence: {confidence}")
    print(f"Extracted attributes: {attrs}")

    if garment_type != "bottom":
        print(f"NOTE: expected garment_type='bottom', got {garment_type!r} -- "
              "check the photo or the model's read before trusting the rest "
              "of this scenario's output.")
    else:
        print("garment_type correctly read as 'bottom'.")
        if "neck_type" not in attrs and "sleeve_type" not in attrs:
            print("Confidence correctly NOT penalized for absent neck_type/"
                  f"sleeve_type on a bottoms item (confidence={confidence}, "
                  "graded only against pattern/silhouette/primary_color + "
                  "garment_type -- see FIELD_APPLICABILITY).")

    print(f"Status message: {result.get('status_message')}")
    if result.get("query"):
        target_category = result["query"].get("target_category")
        print(f"build_complementary_query target_category: {target_category!r} "
              f"(expected 'tops', not the old hardcoded 'bottoms')")
    if result.get("ranked_recommendations"):
        print(f"Recommendations: {[r['name'] for r in result['ranked_recommendations']]}")
    print_trace(result)


def scenario_low_confidence(app):
    print("=" * 60)
    print("SCENARIO 3: low-confidence vision tag -> ask user to re-upload")
    print("=" * 60)

    config = {"configurable": {"thread_id": "demo-lowconf-1"}}
    initial = {
        "image_path": "test_images/blurry_photo.jpg",
        "user_id": "kiran-demo-user",
    }
    result = app.invoke(initial, config)

    print(f"Vision confidence: {result.get('vision_confidence')}")
    print(f"Status message shown to user: {result.get('status_message')}")
    if result.get("vision_confidence", 0) >= 0.65:
        print("NOTE: this photo scored ABOVE 0.65 for real -- the low-"
              "confidence branch did not fire. blurry_photo.jpg may need to "
              "be a genuinely harder photo to read before this scenario "
              "demonstrates anything.")
    print_trace(result)


def scenario_search_failure(app):
    print("=" * 60)
    print("SCENARIO 4: search_brand_inventory fails once, retries, then succeeds")
    print("=" * 60)

    config = {"configurable": {"thread_id": "demo-searchfail-1"}}
    initial = {
        "image_path": "test_images/closet_top.jpg",
        "user_id": "kiran-demo-user",
        "_force_search_failure_once": True,
    }
    result = app.invoke(initial, config)
    if result.get("vision_confidence", 0) < 0.65:
        print("Vision came back low-confidence for this photo on this run -- "
              "search was never reached, so the retry-once path this "
              "scenario is meant to exercise didn't fire. Re-run, or use a "
              "photo confirmed to score above threshold.")
    print_trace(result)


def scenario_zero_results(app):
    print("=" * 60)
    print("SCENARIO 5: zero search results -> explicit message, not a blank card view")
    print("=" * 60)

    config = {"configurable": {"thread_id": "demo-zeroresults-1"}}
    initial = {
        "image_path": "test_images/closet_top.jpg",
        "user_id": "kiran-demo-user",
        "_force_zero_results": True,
    }
    result = app.invoke(initial, config)

    print(f"Ranked recommendations: {result.get('ranked_recommendations')}")
    print(f"Status message shown to user: {result.get('status_message')}")
    print_trace(result)


def scenario_vision_total_failure(app):
    print("=" * 60)
    print("SCENARIO 6: vision_extract fails on both attempts -> graceful stop, no crash")
    print("=" * 60)
    config = {"configurable": {"thread_id": "demo-visionfail-1"}}
    initial = {
        "image_path": "test_images/closet_top.jpg",
        "user_id": "kiran-demo-user",
        "_force_vision_failure": True,
    }
    result = app.invoke(initial, config)
    print(f"Status message shown to user: {result.get('status_message')}")
    print_trace(result)


def scenario_build_query_total_failure(app):
    print("=" * 60)
    print("SCENARIO 7: build_query fails on both attempts -> graceful stop, no crash")
    print("=" * 60)
    config = {"configurable": {"thread_id": "demo-buildqueryfail-1"}}
    initial = {
        "image_path": "test_images/closet_top.jpg",
        "user_id": "kiran-demo-user",
        "_force_build_query_failure": True,
    }
    result = app.invoke(initial, config)
    print(f"Status message shown to user: {result.get('status_message')}")
    print_trace(result)


def scenario_remember_preference(app):
    """
    Exercises the full Step 5 loop: state a preference in chat -> approval
    gate -> Mem0 write -> confirm it's reflected in a LATER, real graph
    invocation (not just a direct function call) via node_get_user_preferences.
    """
    print("=" * 60)
    print("SCENARIO 8: state a dislike -> approve remembering it -> confirm it's reflected later")
    print("=" * 60)

    user_id = "kiran-demo-user"
    stated_preference = "I dont like the color orange"

    config = {"configurable": {"thread_id": "demo-preference-1"}}
    result = app.invoke({"user_message": stated_preference, "user_id": user_id}, config)
    print(f"Confirmation message shown to user: {result.get('status_message')}")

    print("\n[human-in-the-loop] User confirms: yes, remember this preference...")
    app.update_state(config, {
        "user_wants_to_remember_preference": True,
        "approved_preference_text": stated_preference,
    })
    final = app.invoke(None, config)  # resume from the interrupt, executes remember_preference
    print(f"Preference saved: {final.get('preference_saved')}")
    print(f"Status message: {final.get('status_message')}")
    print_trace(final)

    print("\nWaiting for Mem0's write to become readable (eventually consistent)...")
    max_attempts = 6
    delay_seconds = 2
    prefs = None
    for attempt in range(1, max_attempts + 1):
        prefs = get_user_preferences(user_id)
        if "orange" in prefs.get("disliked_colors", []):
            print(f"  Attempt {attempt}: 'orange' now visible in disliked_colors.")
            break
        print(f"  Attempt {attempt}: not visible yet, waiting {delay_seconds}s...")
        time.sleep(delay_seconds)
    else:
        print(f"  NOTE: after {max_attempts} attempts (~{max_attempts * delay_seconds}s), "
              f"'orange' still not visible ({prefs}). Likely Mem0 latency on this "
              "particular run, not a code bug -- rerun this scenario alone to check, "
              "or bump max_attempts/delay_seconds.")

    print("\n[later in the same session] User uploads a photo and asks for recommendations...")
    config2 = {"configurable": {"thread_id": "demo-preference-1-followup"}}
    followup = app.invoke(
        {"image_path": "test_images/closet_top.jpg", "user_id": user_id}, config2,
    )
    reflected_prefs = followup.get("user_preferences", {})
    print(f"user_preferences read by the graph on this later run: {reflected_prefs}")
    if "orange" in reflected_prefs.get("disliked_colors", []):
        print("CONFIRMED: the stated dislike is reflected in a later, independent "
              "graph invocation -- the demo goal is satisfied.")
    else:
        print("NOTE: 'orange' not present in this later invocation's user_preferences -- "
              "check Mem0 timing or the thread_id/user_id wiring before recording.")
    print_trace(followup)


if __name__ == "__main__":
    app = build_graph()
    scenario_happy_path(app)
    scenario_bottoms_photo(app)
    scenario_low_confidence(app)
    scenario_search_failure(app)
    scenario_zero_results(app)
    scenario_vision_total_failure(app)
    scenario_build_query_total_failure(app)
    scenario_remember_preference(app)
