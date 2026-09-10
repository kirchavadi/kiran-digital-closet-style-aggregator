"""
Demo runner for the Digital Closet agent graph.

Runs three scenarios so every documented branch actually executes, which is
exactly what the handout's video demo should show:
  1. Happy path: confident vision tag, enough results, present + save.
  2. Low-confidence vision tag -> re-upload branch (no retrieval attempted).
  3. Forced search-tool failure -> retry-once -> plain-language message.

Usage:
    python demo.py
"""

from graph import build_graph


def print_trace(state):
    print("\n--- trace ---")
    for line in state.get("trace", []):
        print(f"  {line}")
    print("-------------\n")


def scenario_happy_path(app):
    print("=" * 60)
    print("SCENARIO 1: happy path (confident tag, results found, save approved)")
    print("=" * 60)

    config = {"configurable": {"thread_id": "demo-happy-1"}}
    initial = {
        "image_path": "/mnt/user-data/uploads/closet_top.jpg",
        "user_id": "kiran-demo-user",
    }

    # Force a confident vision read for this scenario by monkeypatching is
    # overkill for a demo script -- instead just retry the invoke until we
    # land a confident draw, since vision_extract_attributes randomizes
    # confidence to let this same script show both branches naturally.
    result = app.invoke(initial, config)
    while result["vision_confidence"] < 0.65:
        result = app.invoke(initial, config)

    print(f"Vision confidence: {result['vision_confidence']}")
    if result.get("ranked_recommendations"):
        print(f"Recommendations: {[r['name'] for r in result['ranked_recommendations']]}")
        print(f"Styling note: {result['styling_note']}")

    # Graph paused before save_to_digital_closet (interrupt_before). This is
    # the human-in-the-loop gate. Simulate the user clicking "save" on the
    # first card, then resume the graph.
    print("\n[human-in-the-loop] User clicks 'Save' on the first recommendation...")
    app.update_state(config, {
        "user_wants_to_save": True,
        "approved_item_id": result["ranked_recommendations"][0]["id"],
    })
    final = app.invoke(None, config)  # resume from the interrupt
    print(f"Saved: {final.get('saved')}")
    print_trace(final)


def scenario_low_confidence(app):
    print("=" * 60)
    print("SCENARIO 2: low-confidence vision tag -> ask user to re-upload")
    print("=" * 60)

    config = {"configurable": {"thread_id": "demo-lowconf-1"}}
    initial = {
        "image_path": "/mnt/user-data/uploads/blurry_photo.jpg",
        "user_id": "kiran-demo-user",
    }
    result = app.invoke(initial, config)
    while result["vision_confidence"] >= 0.65:
        # force the low-confidence branch for the demo
        config = {"configurable": {"thread_id": f"demo-lowconf-{result['vision_confidence']}"}}
        result = app.invoke(initial, config)

    print(f"Vision confidence: {result['vision_confidence']}")
    print(f"Status message shown to user: {result.get('status_message')}")
    print_trace(result)


def scenario_search_failure(app):
    print("=" * 60)
    print("SCENARIO 3: search_brand_inventory fails once, retries, then succeeds")
    print("=" * 60)

    config = {"configurable": {"thread_id": "demo-searchfail-1"}}
    initial = {
        "image_path": "/mnt/user-data/uploads/closet_top.jpg",
        "user_id": "kiran-demo-user",
        "_force_search_failure_once": True,
    }
    result = app.invoke(initial, config)
    while result["vision_confidence"] < 0.65:
        result = app.invoke(initial, config)
    print_trace(result)


def scenario_zero_results(app):
    print("=" * 60)
    print("SCENARIO 4: zero search results -> explicit message, not a blank card view")
    print("=" * 60)

    config = {"configurable": {"thread_id": "demo-zeroresults-1"}}
    initial = {
        "image_path": "/mnt/user-data/uploads/closet_top.jpg",
        "user_id": "kiran-demo-user",
        "_force_zero_results": True,
    }
    result = app.invoke(initial, config)
    while result["vision_confidence"] < 0.65:
        result = app.invoke(dict(initial), config)

    print(f"Ranked recommendations: {result.get('ranked_recommendations')}")
    print(f"Status message shown to user: {result.get('status_message')}")
    print_trace(result)


def scenario_vision_total_failure(app):
    print("=" * 60)
    print("SCENARIO 5: vision_extract fails on both attempts -> graceful stop, no crash")
    print("=" * 60)
    config = {"configurable": {"thread_id": "demo-visionfail-1"}}
    initial = {
        "image_path": "/mnt/user-data/uploads/closet_top.jpg",
        "user_id": "kiran-demo-user",
        "_force_vision_failure": True,
    }
    result = app.invoke(initial, config)
    print(f"Status message shown to user: {result.get('status_message')}")
    print_trace(result)


def scenario_build_query_total_failure(app):
    print("=" * 60)
    print("SCENARIO 6: build_query fails on both attempts -> graceful stop, no crash")
    print("=" * 60)
    config = {"configurable": {"thread_id": "demo-buildqueryfail-1"}}
    initial = {
        "image_path": "/mnt/user-data/uploads/closet_top.jpg",
        "user_id": "kiran-demo-user",
        "_force_build_query_failure": True,
    }
    result = app.invoke(initial, config)
    while result.get("vision_confidence") is not None and result["vision_confidence"] < 0.65:
        result = app.invoke(dict(initial), config)
    print(f"Status message shown to user: {result.get('status_message')}")
    print_trace(result)


if __name__ == "__main__":
    app = build_graph()
    scenario_happy_path(app)
    scenario_low_confidence(app)
    scenario_search_failure(app)
    scenario_zero_results(app)
    scenario_vision_total_failure(app)
    scenario_build_query_total_failure(app)
