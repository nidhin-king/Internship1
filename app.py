import streamlit as st
import whisper
import librosa
import pandas as pd
import plotly.express as px
from transformers import pipeline

st.set_page_config(page_title="Voice Emotion Analyzer", layout="wide")

st.title("🎙️ Voice Emotion Analysis Dashboard")

# Load models (cached)
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")
    emotion_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=False
    )
    return whisper_model, emotion_model

whisper_model, emotion_model = load_models()

uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "m4a"])

if uploaded_file:
    st.audio(uploaded_file)

    with st.spinner("Analyzing audio..."):
        # Save file
        audio_path = "temp_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.read())

        # Transcribe with timestamps
        result = whisper_model.transcribe(audio_path, verbose=False)

        records = []
        for seg in result["segments"]:
            text = seg["text"]
            start = seg["start"]
            end = seg["end"]

            emotion = emotion_model(text)[0]
            records.append({
                "Start (sec)": round(start, 2),
                "End (sec)": round(end, 2),
                "Text": text,
                "Emotion": emotion["label"],
                "Confidence": round(emotion["score"], 2)
            })

        df = pd.DataFrame(records)

    st.success("Analysis Completed!")

    # 📊 Dashboard
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Emotion Timeline")
        st.dataframe(df, use_container_width=True)

    with col2:
        st.subheader("📊 Emotion Distribution")
        fig1 = px.pie(df, names="Emotion", title="Overall Emotions")
        st.plotly_chart(fig1, use_container_width=True)

    # ⏱️ Emotion vs Time
    st.subheader("⏱️ Emotion Change Over Time")
    fig2 = px.scatter(
        df,
        x="Start (sec)",
        y="Emotion",
        color="Emotion",
        size="Confidence",
        title="Emotion Detection by Time"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 🔄 Emotion Change Points
    st.subheader("🔄 Emotion Change Moments")
    changes = []
    prev = None

    for _, row in df.iterrows():
        if prev and prev != row["Emotion"]:
            changes.append({
                "Time (sec)": row["Start (sec)"],
                "Emotion Changed To": row["Emotion"]
            })
        prev = row["Emotion"]

    st.table(pd.DataFrame(changes))