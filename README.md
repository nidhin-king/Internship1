# 🎤 Voice Emotion AI Dashboard

A premium AI-powered dashboard for analyzing emotions in voice recordings. Built with Streamlit, Librosa, and Scikit-Learn.

## 🚀 Features
- **Real-time Emotion Analysis**: Upload WAV or MP3 files to detect emotions.
- **Timeline View**: See how emotions change throughout the recording.
- **Distribution Analytics**: Visualize the frequency of detected emotions.
- **One-Click Training**: Re-train the model directly from the UI.
- **RAVDESS Support**: Automatically parses the RAVDESS dataset structure.

## 🛠 Setup & Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App**:
   ```bash
   streamlit run app.py
   ```

3. **Train the Model**:
   If the model is not present, the app will prompt you to train it. Alternatively, run:
   ```bash
   python train_model.py
   ```

## 📊 Dataset Structure
The app expects a `dataset` folder containing RAVDESS filename-formatted `.wav` files.
Filename format: `MM-AA-EE-SS-CC-II-AA.wav` where `EE` is the emotion code.

| Code | Emotion |
|------|---------|
| 01 | Neutral |
| 02 | Calm |
| 03 | Happy |
| 04 | Sad |
| 05 | Angry |
| 06 | Fearful |
| 07 | Disgust |
| 08 | Surprised |

## 🏗 Technology Stack
- **Dashboard**: [Streamlit](https://streamlit.io/)
- **Audio Processing**: [Librosa](https://librosa.org/)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/) (Random Forest)
- **Serialization**: [Joblib](https://joblib.readthedocs.io/)
