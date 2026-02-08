import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Voice Emotion Analysis", layout="wide")

st.title("🎤 Voice Emotion Analysis Dashboard")

uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3"]
)

if uploaded_file is not None:
    # Load audio safely (NO ffmpeg needed)
    y, sr = librosa.load(uploaded_file, sr=None)

    duration = len(y) / sr

    st.success("Audio loaded successfully")
    st.write(f"Sample Rate: {sr}")
    st.write(f"Duration: {round(duration, 2)} seconds")

    # Dummy emotion detection (placeholder)
    time = np.linspace(0, duration, num=10)
    emotions = ["Neutral", "Happy", "Sad", "Angry"]
    detected = np.random.choice(emotions, size=len(time))

    st.subheader("Emotion Timeline")
    for t, e in zip(time, detected):
        st.write(f"⏱ {round(t,2)} sec → {e}")

    # Chart
    emotion_count = {e: list(detected).count(e) for e in emotions}

    fig, ax = plt.subplots()
    ax.bar(emotion_count.keys(), emotion_count.values())
    ax.set_title("Emotion Distribution")
    st.pyplot(fig)
