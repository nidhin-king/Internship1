import librosa
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

DATASET_PATH = "dataset"

# RAVDESS Emotion mapping
EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}. Error: {e}")
        return None

X = []
y = []

print("Starting feature extraction...")
for root, dirs, files in os.walk(DATASET_PATH):
    for file in tqdm(files):
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            
            # RAVDESS filename format: 03-01-06-01-02-01-12.wav
            # Emotion is the 3rd index (06 in this example)
            parts = file.split("-")
            if len(parts) >= 3:
                emotion_code = parts[2]
                emotion_label = EMOTIONS.get(emotion_code)
                
                if emotion_label:
                    feature = extract_features(file_path)
                    if feature is not None:
                        X.append(feature)
                        y.append(emotion_label)

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    print("No data found! Please check your dataset path and structure.")
    exit()

print(f"Extraction complete. Total samples: {len(X)}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save label encoder
joblib.dump(le, "label_encoder.pkl")
print("Label encoder saved.")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Train model
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"Model Accuracy: {score * 100:.2f}%")

# Save model
joblib.dump(model, "emotion_model.pkl")
print("Model saved as emotion_model.pkl")

print("Training finished successfully!")
