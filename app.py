import streamlit as st
from streamlit_webrtc import webrtc_streamer

from emotion_processor import EmotionProcessor

st.set_page_config(page_title="Emotion Music App")

st.title("🎵 Emotion Based Music Recommendation System")

# ---------------- Session State ----------------
if "emotion" not in st.session_state:
    st.session_state["emotion"] = ""

# ---------------- Inputs ----------------
lang = st.text_input("Enter Language (e.g. Hindi, English)")
singer = st.text_input("Enter Singer (optional)")

# ---------------- Webcam ----------------
if lang:
    webrtc_streamer(
        key="emotion-detection",
        video_processor_factory=EmotionProcessor,
        desired_playing_state=True
    )

# ---------------- Recommendation ----------------
if st.button("Recommend Songs 🎶"):

    emotion = st.session_state.get("emotion", "")

    if not emotion:
        st.warning("Please allow camera to detect your emotion first.")
    else:
        query = f"{lang}+{emotion}+song+{singer}"
        url = f"https://www.youtube.com/results?search_query={query}"

        st.success("Here are your recommended songs 🎧")
        st.markdown(f"[👉 Open YouTube Results]({url})")

        st.session_state["emotion"] = ""
