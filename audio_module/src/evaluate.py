# src/evaluate.py
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = "./model/audio_lstm_model.h5"
ENCODER_PATH = "./model/audio_label_encoder.pkl"
DATA_PATH = "./data/processed/audio_mental_health_features.npz"


def evaluate():
    print("📥 Loading model...")
    model = load_model(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    print("📥 Loading dataset...")
    data = np.load(DATA_PATH, allow_pickle=True)
    X = data["X"]
    y = data["y"]

    y_true = encoder.transform(y)

    print("🔍 Predicting...")
    y_pred_prob = model.predict(X)
    y_pred = np.argmax(y_pred_prob, axis=1)

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=encoder.classes_))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
                xticklabels=encoder.classes_,
                yticklabels=encoder.classes_)
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    evaluate()
