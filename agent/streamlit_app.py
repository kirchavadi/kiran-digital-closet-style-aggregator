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
import re
import uuid

import streamlit as st

from graph import build_graph
from tools import _CLOSET_DIR as CLOSET_DATA_DIR, get_saved_closet_items, get_user_preferences

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
        "processing": False,
        "pending_invoke": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _list_known_profiles() -> list:
    # Profiles = user_ids that already have a saved-closet JSON file, plus
    # whichever user_id is currently selected (so a brand-new profile name
    # you just typed doesn't vanish from the list before its first save).
    # Reuses tools.py's own closet-directory constant (_CLOSET_DIR) rather
    # than reconstructing the path, so this can never drift from where
    # save_to_digital_closet actually writes.
    profiles = set()
    if os.path.isdir(CLOSET_DATA_DIR):
        for fname in os.listdir(CLOSET_DATA_DIR):
            if fname.endswith(".json"):
                profiles.add(fname[:-len(".json")])
    profiles.add(st.session_state.get("user_id", "kiran-demo-user"))
    return sorted(profiles)


def _normalize_profile_name(raw: str) -> str:
    # Lowercase letters, digits, "_", "." and "-" only -- the same character set
    # tools.py's _closet_file_path allows, so the profile name, the closet
    # filename and the Mem0 user_id are always identical (no silent "Kiran C"
    # vs "Kiran_C" drift).
    name = re.sub(r"[^a-z0-9_.-]+", "-", raw.strip().lower()).strip("-._")
    return name[:40].strip("-._")


def _create_profile():
    # form_submit_button on_click callback. Callbacks run BEFORE the script
    # reruns, which is the only moment Streamlit allows changing a widget's own
    # session-state value (here: the Profile dropdown, so it lands on the new
    # profile). Writing an empty closet file is what makes the profile
    # "exist": _list_known_profiles() finds it on every later run, so the new
    # profile stays in the dropdown even before its first saved item.
    name = _normalize_profile_name(st.session_state.get("new_profile_name", ""))
    if not name:
        st.session_state["profile_notice"] = ("warning", "Type a profile name first.")
        return
    os.makedirs(CLOSET_DATA_DIR, exist_ok=True)
    path = os.path.join(CLOSET_DATA_DIR, f"{name}.json")
    already_existed = os.path.exists(path)
    if not already_existed:
        with open(path, "w") as f:
            f.write("[]")
    st.session_state["user_id"] = name
    st.session_state["profile_choice"] = name
    st.session_state["new_profile_name"] = ""
    st.session_state["profile_notice"] = (
        "info" if already_existed else "success",
        f"Switched to existing profile: {name}" if already_existed
        else f"Created profile: {name}",
    )


_init_session_state()
app = get_app()

NEW_PROFILE_OPTION = "+ New profile..."

with st.sidebar:
    st.header("\U0001F457 Digital Closet")
    processing = st.session_state.get("processing", False)

    profile_options = _list_known_profiles() + [NEW_PROFILE_OPTION]
    # The dropdown's value lives in session state under key="profile_choice"
    # (stable widget identity; also lets _create_profile move the dropdown to
    # the new profile). Make sure the stored value is still a valid option.
    if st.session_state.get("profile_choice") not in profile_options:
        current = st.session_state["user_id"]
        st.session_state["profile_choice"] = (
            current if current in profile_options else profile_options[0]
        )
    chosen_profile = st.selectbox(
        "Profile", profile_options, key="profile_choice", disabled=processing
    )

    notice = st.session_state.pop("profile_notice", None)
    if notice:
        getattr(st, notice[0])(notice[1])

    if chosen_profile == NEW_PROFILE_OPTION:
        # A profile is only created when Create is pressed (or Enter, since this
        # is a form). Until then the app keeps using the current profile.
        with st.form("new_profile_form", border=False):
            st.text_input(
                "New profile name", placeholder="e.g. jordan",
                key="new_profile_name", disabled=processing,
            )
            st.form_submit_button(
                "Create profile", on_click=_create_profile, disabled=processing
            )
        st.caption(f"Still using profile: {st.session_state['user_id']}")
    else:
        st.session_state["user_id"] = chosen_profile

    page = st.radio(
        "Go to", ["Upload & Recommend", "My Closet", "Preferences"], disabled=processing
    )
    if processing:
        st.caption("\u23f3 Processing your last upload -- navigation is paused until it finishes.")
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
            st.image(item["image_url"], width="stretch")
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

    processing = st.session_state.get("processing", False)

    uploaded_file = st.file_uploader(
        "Upload a photo", type=["jpg", "jpeg", "png"], disabled=processing
    )

    if uploaded_file is not None and not processing:
        sig = f"{uploaded_file.name}:{uploaded_file.size}"
        if sig != st.session_state["uploaded_file_sig"]:
            # A genuinely new upload (not just a rerun triggered by some
            # other widget) -- save it, then IMMEDIATELY set processing=True
            # and rerun BEFORE calling the graph. This is deliberate: Streamlit
            # reruns the whole script on every widget interaction, and a new
            # interaction (like clicking a different sidebar page) cancels
            # whatever script run is currently in flight -- including a
            # blocking app.invoke() call. If we called invoke() inline in
            # THIS run, the sidebar the user sees was already sent to the
            # browser enabled (it was drawn earlier in this same run, before
            # we knew we needed to lock it), so clicking away mid-upload
            # would silently kill the request and lose the result. Rerunning
            # first means the NEXT run's sidebar is drawn with navigation
            # disabled from the start, before the slow call ever begins.
            image_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{uploaded_file.name}")
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            thread_id = f"streamlit-{st.session_state['user_id']}-{uuid.uuid4().hex[:8]}"
            st.session_state["uploaded_file_sig"] = sig
            st.session_state["uploaded_image_path"] = image_path
            st.session_state["upload_thread_id"] = thread_id
            st.session_state["upload_saved_item_id"] = None
            st.session_state["pending_invoke"] = {
                "image_path": image_path,
                "user_id": st.session_state["user_id"],
                "thread_id": thread_id,
            }
            st.session_state["processing"] = True
            st.rerun()

    if st.session_state.get("processing") and st.session_state.get("pending_invoke"):
        pending = st.session_state["pending_invoke"]
        config = {"configurable": {"thread_id": pending["thread_id"]}}
        initial = {"image_path": pending["image_path"], "user_id": pending["user_id"]}

        status_placeholder = st.empty()
        with status_placeholder.container():
            _, center_col, _ = st.columns([1, 2, 1])
            with center_col:
                st.markdown(
                    """
                    <div style="
                        text-align:center;
                        padding:1.25rem 1rem;
                        border-radius:12px;
                        border:2px solid #FF8C00;
                        background-color:rgba(255,140,0,0.15);
                    ">
                        <div style="font-size:1.4rem; font-weight:700;">
                            🔎 Reading the photo and finding pairings...
                        </div>
                        <div style="font-size:0.95rem; margin-top:0.4rem; opacity:0.85;">
                            This takes a few seconds -- please wait. Navigation is
                            locked until this finishes so your upload can't be
                            interrupted.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        try:
            result = app.invoke(initial, config)
            st.session_state["upload_result"] = result
        finally:
            st.session_state["processing"] = False
            st.session_state["pending_invoke"] = None
        status_placeholder.empty()
        st.rerun()

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
                st.image(item["image_url"], width="stretch")
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

    current = get_user_preferences(st.session_state["user_id"])
    with st.container(border=True):
        st.caption(f"Currently on file for profile: {st.session_state['user_id']}")
        disliked = current.get("disliked_colors") or []
        brands = current.get("preferred_brands") or []
        budget = current.get("budget_max")
        st.write(f"Disliked colors: {', '.join(disliked) if disliked else 'none'}")
        st.write(f"Preferred brands: {', '.join(brands) if brands else 'none'}")
        if budget is not None:
            st.write(f"Budget max: ${budget:,.2f}")

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
