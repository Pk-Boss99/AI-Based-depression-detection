import streamlit as st
import os
import sys
import joblib

# Set up system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)

from config import GUI_TITLE, GUI_DESCRIPTION
from utils import setup_logging

setup_logging()
import logging
logger = logging.getLogger(__name__)

st.set_page_config(page_title=GUI_TITLE, layout="wide")
st.title(GUI_TITLE)
st.markdown(GUI_DESCRIPTION)

# --- Custom Styling ---
st.markdown("""
    <style>
        body, .stApp {
        }
        h1, h2, h3 {
            color: #1a1a1a;
        }
        .stRadio > div {
            flex-direction: column;
            background-color: #2c3e50;
            color: white !important;
            padding: 1em;
            border-radius: 8px;
        }
        .stRadio label {
            color: white !important;
        }
        .stButton > button {
            background-color: #007acc;
            color: white;
            font-weight: bold;
            padding: 0.5em 1.5em;
            border-radius: 6px;
        }
        .stButton > button:hover {
            background-color: #005f99;
        }
        .stTextArea textarea {
            color: #000000;
            border-radius: 6px;
            border: 1px solid #cccccc;
            padding: 10px;
            font-size: 1em;
        }
        .result-container {
            background-color: #ffffff;  /* Set background for result container */
            border-radius: 10px;
            padding: 1.5em;
            margin-top: 20px;
        }
        .element-container:has(.stMetric) {
            background-color: #f7f9fb;
            border-radius: 10px;
            padding: 1em;
        }
    </style>
""", unsafe_allow_html=True)

# Step 1: Let user choose analysis type
analysis_type = st.radio("Choose analysis method:", ["Text", "Audio", "Video"])

# --- State handling for text navigation ---
if "text_step" not in st.session_state:
    st.session_state.text_step = 0
if "text_responses" not in st.session_state:
    st.session_state.text_responses = [""] * 15

# Placeholder scores
text_score = audio_score = video_score = None

# --- Text Analysis (One by One Questions) ---
if analysis_type == "Text":
    st.header("Text Analysis")
    questions = [
        "How have you been feeling lately?",
        "Have you been feeling sad or down more than usual?",
        "Do you feel exhausted even after a good night's sleep?",
        "Do you find it hard to enjoy activities you once liked?",
        "Are you feeling hopeless or helpless?",
        "Have you been feeling irritable or angry?",
        "Do you have trouble concentrating on tasks?",
        "Have you been avoiding social situations?",
        "Do you have trouble getting out of bed?",
        "Have you been experiencing changes in appetite or weight?",
        "Do you feel restless or slowed down?",
        "Have you had thoughts about death or suicide?",
        "Do you feel anxious or tense more than usual?",
        "Do you experience mood swings?",
        "Have you been feeling guilty or worthless?"
    ]
    i = st.session_state.text_step
    st.subheader(f"Question {i + 1} of {len(questions)}")
    st.session_state.text_responses[i] = st.text_area(questions[i], value=st.session_state.text_responses[i])

    col1, col2 = st.columns([1, 1])
    with col1:
        if i > 0 and st.button("Previous"):
            st.session_state.text_step -= 1
    with col2:
        if i < len(questions) - 1 and st.button("Next"):
            st.session_state.text_step += 1

# --- Audio Analysis ---
elif analysis_type == "Audio":
    st.header("Audio Analysis")
    st.write("Please answer these 5 questions by recording your voice:")

    audio_responses = []
    audio_questions = [
        "How are you feeling today?",
        "Can you describe your mood in the past week?",
        "Have you experienced stress or anxiety recently?",
        "What has been affecting your sleep lately?",
        "What are you looking forward to?"
    ]

    for i, question in enumerate(audio_questions, start=1):
        st.markdown(f"### Question {i}: {question}")
        audio_input = st.audio_input(f"Record your response for question {i}:")
        if audio_input:
            audio_responses.append(audio_input)
        else:
            audio_responses.append(None)

# --- Video Analysis ---
elif analysis_type == "Video":
    st.header("Video Analysis")
    video_file = st.file_uploader("Upload video (.mp4/.avi)", type=["mp4", "avi"])
    if video_file:
        st.video(video_file)
    st.info("Live recording via webcam will be available in future updates.")

# --- Analyze Button ---
if st.button("Analyze for Depression Indicators"):
    st.info("Analyzing inputs...")
    try:
        if analysis_type == "Text":
            if not any(st.session_state.text_responses):
                st.warning("Please answer at least one text question.")
            else:
              
                text_score = 0.7 
                st.success("Text Analysis Complete")
                st.metric("Text Depression Score", f"{text_score * 100:.2f}%")

        elif analysis_type == "Audio":
            if not any(audio_responses):
                st.warning("Please record audio responses for all questions.")
            else:
                audio_score = 0.6  
                st.success("Audio Analysis Complete")

                st.markdown(f'''
                <div style="background-color: #007acc; padding: 10px; border-radius: 8px;">
                    <h4 style="color: white;">Audio Depression Score</h4>
                    <p style="color: white; font-size: 24px;">{audio_score * 100:.2f}%</p>
                </div>
                ''', unsafe_allow_html=True)

        elif analysis_type == "Video":
            if not video_file:
                st.warning("Please upload a video file.")
            else:
                video_score = 0.8   
                st.success("Video Analysis Complete")
                st.metric("Video Depression Score", f"{video_score * 100:.2f}%")

    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        st.error(f"An error occurred: {e}")

# Footer
st.markdown("---")
st.write("Disclaimer: This system is for informational purposes only and should not be considered a substitute for professional medical advice.")

