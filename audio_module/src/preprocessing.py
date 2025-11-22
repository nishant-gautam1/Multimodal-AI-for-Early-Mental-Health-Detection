import os
import numpy as np
from tqdm import tqdm
from audio_utils import extract_mfcc

RAVDESS_PATH = "./data/RAVDESS/"
TESS_PATH = "./data/TESS/"
OUT_PATH = "./data/processed/audio_mental_health_features.npz"

# Emotion mappings (same as working script)
ravdess_emotions = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def map_to_mental_health(e):
    if e is None:
        return None
    e = e.lower()
    if e in ['happy', 'calm', 'pleasant', 'neutral']:
        return 'normal'
    if e in ['sad', 'fearful', 'disgust']:
        return 'depressed'
    if e in ['angry', 'surprised']:
        return 'stressed'
    return None

def preprocess():
    features, labels = [], []

    print("🎧 Extracting RAVDESS features...")
    for actor in tqdm(os.listdir(RAVDESS_PATH)):
        actor_folder = os.path.join(RAVDESS_PATH, actor)
        if not os.path.isdir(actor_folder):
            continue
        for file in os.listdir(actor_folder):
            if file.endswith(".wav"):
                parts = file.split("-")
                emotion_code = parts[2]
                emotion = ravdess_emotions.get(emotion_code)
                mental_state = map_to_mental_health(emotion)

                if mental_state is None:
                    continue

                mfcc = extract_mfcc(os.path.join(actor_folder, file))
                if mfcc is not None:
                    features.append(mfcc)
                    labels.append(mental_state)

    print("🎙 Extracting TESS features...")
    tess_emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant', 'sad', 'surprise']

    for root, _, files in os.walk(TESS_PATH):
        for file in files:
            if file.endswith(".wav"):
                emo_found = None
                for emo in tess_emotions:
                    if emo in file.lower():
                        emo_found = emo
                        break
                if emo_found is None:
                    continue

                mental_state = map_to_mental_health(emo_found)
                if mental_state is None:
                    continue

                mfcc = extract_mfcc(os.path.join(root, file))
                if mfcc is not None:
                    features.append(mfcc)
                    labels.append(mental_state)

    X = np.array(features)
    y = np.array(labels)

    os.makedirs("./data/processed/", exist_ok=True)
    np.savez(OUT_PATH, X=X, y=y)
    print("✅ Saved preprocessed dataset at:", OUT_PATH)

if __name__ == "__main__":
    preprocess()
