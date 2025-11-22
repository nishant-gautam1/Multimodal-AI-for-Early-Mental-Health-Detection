import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

DATA_PATH = "./data/processed/audio_mental_health_features.npz"
MODEL_PATH = "./model/audio_lstm_model.h5"
ENCODER_PATH = "./model/audio_label_encoder.pkl"

MAX_LEN = 216

def train():
    print("📥 Loading dataset...")
    data = np.load(DATA_PATH, allow_pickle=True)

    X = data["X"]
    y = data["y"]

    print("🔤 Encoding labels...")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_cat = to_categorical(y_enc)

    num_classes = y_cat.shape[1]

    os.makedirs("./model", exist_ok=True)
    joblib.dump(le, ENCODER_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42
    )

    print("🤖 Building LSTM model...")
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(40, MAX_LEN)),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    print("🚀 Training started...")
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

    model.save(MODEL_PATH)
    print("✅ Model saved:", MODEL_PATH)


if __name__ == "__main__":
    train()
