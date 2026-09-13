"""
Streamlit UI for Kiran's Digital Closet agent (Week 3 Project).

Thin UI in front of build_graph() -- per README.md's step 4 checklist item:
"Put a thin Streamlit UI in front of build_graph() for the demo video."
No new backend logic lives here; every button click just drives the same
graph.invoke() / update_state() / invoke(None, ...) pattern demo.py already
proves works.

Three pages, one file:
  1. Upload & Recommend -- the core agentic flow (photo upload -> vision-tag
     -> retrieval -> ranking -> present cards -> human-approved save).
  2. My Closet -- read-only view of get_saved_closet_items(user_id), backed
     by the flat JSON-per-user file save_to_digital_closet now writes to
     (Sept 13 2026 decision, option b -- see
     claude/session7_closet_json_persistence_design.md). Only exists
     because that decision was made; if (a) had been chosen instead, this
     page would simply be deleted.
  3. Preferences -- optional, demonstrates Step 5's remember_user_preference
     write path (same human-approval gate as save) so the demo video can
     show both gated write tools, not just one.

Placement: this file goes in the SAME directory as graph.py / tools.py /
state.py (i.e. agent/streamlit_app.py), so its bare imports
(`from graph import build_graph`, `from tools import get_saved_closet_items`)
resolve the same way demo.py's already do.

Run:
    cd agent && streamlit run streamlit_app.py

Requires the same environment variables demo.py already needs
(FIREWORKS_API_KEY, NEBIUS_API_KEY, PINECONE_API_KEY, MEM0_API_KEY) and the
`streamlit` package (pip install streamlit).
"""

import os
import uuid

import streamlit as st

from graph import build_graph
from tools import get_saved_closet_items

st.set_page_config(page_title="Kiran's Digital Closet", page_icon="\U0001F457", layout="wide")

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@st.cache_resource
def get_app():
    # Cached once per Streamlit server process. The MemorySaver checkpointer
    # inside build_graph() lives as long as this cached object does -- that
    # is what lets update_state() + invoke(None, ...) resume a paused
    # thread_id across Streamlit reruns (every widget click reruns this
    # whole script from the top).
    return build_graph()


def _init_session_state():
    defaults = {
        "user_id": "kiran-demo-user",
        "upload_thread_id": None,
        "upload_result": None,
        "uploaded_file_sig": None,
        "uploaded_image_path": None,
        "upload_saved_item_id": None,
        "pref_thread_id": None,
        "pref_result": None,
        "pref_text": "",
        "pref_stage": "idle",  # idle -> awaiting_approval -> done / failed
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()
app = get_app()

with st.sidebar:
    st.header("\U0001F457 Digital Closet")
    st.session_state["user_id"] = st.text_input("User ID", value=st.session_state["user_id"])
    page = st.radio("Go to", ["Upload & Recommend", "My Closet", "Preferences"])
    st.caption(
        "Same agent graph as demo.py -- this UI just drives build_graph() "
        "from widgets instead of a script."
    )


def render_candidate_card(item: dict, container):
    with container:
        st.markdown(f"**{item.get('name', 'Unknown item')}**")
        if item.get("brand"):
            st.caption(item["brand"])
        if item.get("display_mode") == "photo" and item.get("image_url"):
            st.image(item["image_url"], use_container_width=True)
        else:
            st.caption("(no image available for this item)")
        price = item.get("price")
        if price is not None:
            st.write(f"${price:,.2f}")
        if item.get("product_url"):
            st.link_button("View on brand site", item["product_url"])

        if st.session_state["upload_saved_item_id"] == item.get("id"):
            st.success("✓ Saved to your closet")
        elif st.session_state["upload_saved_item_id"] is not None:
            # A different card from this same batch was already saved --
            # keep this demo's write path to one save per upload, same as
            # the interrupt_before gate only ever expecting one approval
            # per present_cards pause.
            st.caption("Upload a new photo to save another item.")
        elif st.button("Save to my closet", key=f"save_{item.get('id')}"):
            config = {"configurable": {"thread_id": st.session_state["upload_thread_id"]}}
            app.update_state(config, {
                "user_wants_to_save": True,
                "approved_item_id": item.get("id"),
            })
            final = app.invoke(None, config)
            st.session_state["upload_result"] = final
            if final.get("saved"):
                st.session_state["upload_saved_item_id"] = item.get("id")
            else:
                st.error(final.get("status_message") or "The save didn't go through.")
            st.rerun()


def page_upload_and_recommend():
    st.title("Upload & Recommend")
    st.write(
        "Upload a photo of one item from your closet -- the agent reads it, "
        "searches the catalog for a complementary piece, and shows you cards "
        "to browse. Saving requires your approval; buying always sends you "
        "to the brand's own site."
    )

    uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        sig = f"{uploaded_file.name}:{uploaded_file.size}"
        if sig != st.session_state["uploaded_file_sig"]:
            # A genuinely new upload (not just a rerun triggered by some
            # other widget) -- save it and start a fresh graph thread.
            image_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{uploaded_file.name}")
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            thread_id = f"streamlit-{st.session_state['user_id']}-{uuid.uuid4().hex[:8]}"
            config = {"configurable": {"thread_id": thread_id}}
            initial = {"image_path": image_path, "user_id": st.session_state["user_id"]}

            with st.spinner("Reading the photo and finding pairings..."):
                result = app.invoke(initial, config)

            st.session_state["uploaded_file_sig"] = sig
            st.session_state["uploaded_image_path"] = image_path
            st.session_state["upload_thread_id"] = thread_id
            st.session_state["upload_result"] = result
            st.session_state["upload_saved_item_id"] = None

    result = st.session_state["upload_result"]
    if result is None:
        return

    st.divider()

    if st.session_state.get("uploaded_image_path"):
        img_col, _ = st.columns([1, 3])
        with img_col:
            st.image(
                st.session_state["uploaded_image_path"],
                caption="Your uploaded item",
                width="stretch",
            )

    confidence = result.get("vision_confidence")
    if confidence is not None:
        st.caption(f"Vision confidence: {confidence}")

    st.subheader("Recommendations")

    if result.get("status_message") and not result.get("ranked_recommendations"):
        st.warning(result["status_message"])
        with st.expander("Agent trace"):
            for line in result.get("trace", []):
                st.text(line)
        return

    if result.get("styling_note"):
        st.info(result["styling_note"])

    recommendations = result.get("ranked_recommendations") or []
    if not recommendations:
        st.warning("No recommendations to show for this photo.")
    else:
        cols = st.columns(min(3, len(recommendations)))
        for i, item in enumerate(recommendations):
            render_candidate_card(item, cols[i % len(cols)])

    with st.expander("Agent trace (for grading transparency)"):
        for line in result.get("trace", []):
            st.text(line)


def page_my_closet():
    st.title("My Closet")
    items = get_saved_closet_items(st.session_state["user_id"])
    if not items:
        st.info("Nothing saved yet -- approve a save on the Upload & Recommend page.")
        return
    cols = st.columns(3)
    for i, item in enumerate(reversed(items)):  # most recently saved first
        with cols[i % 3]:
            st.markdown(f"**{item.get('name', 'Unknown item')}**")
            if item.get("brand"):
                st.caption(item["brand"])
            if item.get("display_mode") == "photo" and item.get("image_url"):
                st.image(item["image_url"], use_container_width=True)
            price = item.get("price")
            if price is not None:
                st.write(f"${price:,.2f}")
            if item.get("product_url"):
                st.link_button("View on brand site", item["product_url"])
            saved_at = item.get("saved_at")
            if saved_at:
                st.caption(f"Saved {saved_at}")
            st.divider()


def page_preferences():
    st.title("Preferences")
    st.write(
        "Tell the assistant about a style preference -- a disliked color, a "
        "favorite brand, a budget. It asks for your approval before "
        "remembering anything, using the same human-approval gate as saving "
        "an item."
    )

    text = st.text_input(
        "Tell it something",
        value=st.session_state["pref_text"],
        placeholder="e.g. I don't like the color orange",
    )

    if st.button("Submit") and text.strip():
        thread_id = f"streamlit-pref-{st.session_state['user_id']}-{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}
        result = app.invoke(
            {"user_message": text, "user_id": st.session_state["user_id"]}, config
        )
        st.session_state["pref_thread_id"] = thread_id
        st.session_state["pref_result"] = result
        st.session_state["pref_text"] = text
        st.session_state["pref_stage"] = "awaiting_approval"

    result = st.session_state["pref_result"]

    if st.session_state["pref_stage"] == "awaiting_approval" and result:
        st.write(result.get("status_message"))
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, remember this"):
                config = {"configurable": {"thread_id": st.session_state["pref_thread_id"]}}
                app.update_state(config, {
                    "user_wants_to_remember_preference": True,
                    "approved_preference_text": st.session_state["pref_text"],
                })
                final = app.invoke(None, config)
                st.session_state["pref_result"] = final
                st.session_state["pref_stage"] = "done" if final.get("preference_saved") else "failed"
                st.rerun()
        with col2:
            if st.button("No, never mind"):
                st.session_state["pref_stage"] = "idle"
                st.session_state["pref_text"] = ""
                st.rerun()
    elif st.session_state["pref_stage"] == "done" and result:
        st.success(result.get("status_message") or "Got it -- remembered.")
    elif st.session_state["pref_stage"] == "failed" and result:
        st.error(result.get("status_message") or "Couldn't save that preference.")


if page == "Upload & Recommend":
    page_upload_and_recommend()
elif page == "My Closet":
    page_my_closet()
else:
    page_preferences()
