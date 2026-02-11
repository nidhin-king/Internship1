import streamlit as st
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import subprocess
import time

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="Voice Emotion AI", layout="wide", page_icon="🎤")

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1 {
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎤 Voice Emotion AI Dashboard")

# -------------------------------
# Model Management
# -------------------------------

MODEL_FILE = "emotion_model.pkl"
ENCODER_FILE = "label_encoder.pkl"

@st.cache_resource
def load_models():
    if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
        model = joblib.load(MODEL_FILE)
        le = joblib.load(ENCODER_FILE)
        return model, le
    return None, None

model, le = load_models()

if model is None:
    st.warning("⚠️ Model not found! Please train the model first.")
    if st.button("🚀 Train Model Now"):
        with st.spinner("Training model on dataset... This may take a few minutes."):
            try:
                # Run the training script
                result = subprocess.run(["python", "train_model.py"], capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("✅ Model trained successfully!")
                    st.toast("Reloading models...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"❌ Training failed: {result.stderr}")
            except Exception as e:
                st.error(f"❌ Error starting training: {e}")
    st.info("Ensure your 'dataset' folder is populated with RAVDESS files.")
    st.stop()

# -------------------------------
# Feature Extraction
# -------------------------------

def extract_features(audio_chunk, sr):
    mfcc = librosa.feature.mfcc(y=audio_chunk, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

def split_audio(y, sr, duration=3):
    samples_per_chunk = duration * sr
    chunks = []
    timestamps = []
    for i in range(0, len(y), samples_per_chunk):
        chunk = y[i:i + samples_per_chunk]
        if len(chunk) == samples_per_chunk:
            chunks.append(chunk)
            timestamps.append(i / sr)
    return chunks, timestamps

# -------------------------------
# UI Layout
# -------------------------------

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📁 Upload & Settings")
    uploaded_file = st.file_uploader("Upload an audio file (WAV/MP3)", type=["wav", "mp3"])
    chunk_size = st.slider("Analysis Chunk Size (seconds)", 1, 5, 3)

if uploaded_file is not None:
    with st.spinner("Analyzing audio..."):
        y, sr = librosa.load(uploaded_file, sr=None)
        duration = len(y) / sr
        
        chunks, timestamps = split_audio(y, sr, duration=chunk_size)
        
        detected_emotions = []
        for chunk in chunks:
            features = extract_features(chunk, sr).reshape(1, -1)
            prediction = model.predict(features)
            emotion = le.inverse_transform(prediction)[0]
            detected_emotions.append(emotion)

    with col2:
        st.subheader("📈 Analysis Results")
        
        # Summary Metrics
        m1, m2 = st.columns(2)
        m1.metric("Duration", f"{duration:.2f}s")
        if detected_emotions:
            final_emotion = max(set(detected_emotions), key=detected_emotions.count)
            m2.metric("Dominant Emotion", final_emotion.title())
        
        st.divider()

        # Emotion distribution chart
        st.write("#### Emotion Frequency Distribution")
        emotion_counts = {e: detected_emotions.count(e) for e in set(detected_emotions)}
        st.bar_chart(emotion_counts)

    st.divider()
    
    # Timeline
    st.subheader("⏱ Detailed Timeline")
    timeline_cols = st.columns(4)
    for i, (t, e) in enumerate(zip(timestamps, detected_emotions)):
        with timeline_cols[i % 4]:
            mins, secs = divmod(int(t), 60)
            st.markdown(f"**{mins:02d}:{secs:02d}** → `{e}`")

else:
    st.info("Please upload an audio file to begin analysis.")
