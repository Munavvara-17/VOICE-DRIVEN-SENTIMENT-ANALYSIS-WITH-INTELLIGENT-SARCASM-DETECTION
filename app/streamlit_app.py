import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import tempfile
import os
from backend.audio.speech_to_text import speech_to_text
from backend.utils.preprocess import preprocess_text
from backend.sentiment.sentiment_classifier import classify_sentiment
from backend.sarcasm.sarcasm_detector import detect_sarcasm

st.set_page_config(page_title="Voice Sentiment & Sarcasm", layout="centered")
st.title("🎙️ Voice Sentiment + Sarcasm Detection")

st.markdown("Upload a `.wav` file or record your voice. We'll analyze the sentiment and detect sarcasm.")

# --- Audio Upload ---
uploaded_file = st.file_uploader("📁 Upload a `.wav` file", type=["wav"])

# --- Microphone Button ---
mic_btn = st.button("🎤 Use Microphone")

# --- Speech-to-Text + Analysis ---
text = ""

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(uploaded_file.read())
        temp_audio_path = temp_audio.name

    st.success("Audio uploaded successfully!")
    text = speech_to_text(temp_audio_path)

elif mic_btn:
    st.info("Recording... Please speak clearly.")
    text = speech_to_text()

if text:
    st.markdown("---")
    st.write("**Transcription:**", text)

    cleaned_text = preprocess_text(text)
    sentiment = classify_sentiment(cleaned_text)
    sarcasm = detect_sarcasm(text)

    st.subheader("Results")
    st.write("**Cleaned Text:**", cleaned_text)
    st.write("**Sentiment:**", sentiment)
    st.write("**Sarcasm Detected:**", "Yes" if sarcasm else "No")

    if sarcasm and sentiment == "Positive":
        adjusted_sentiment = "Negative (due to sarcasm)"
    elif sarcasm and sentiment == "Negative":
        adjusted_sentiment = "Positive (due to sarcasm)"
    else:
        adjusted_sentiment = sentiment

    st.success(f"**Final Interpretation:** {adjusted_sentiment}")