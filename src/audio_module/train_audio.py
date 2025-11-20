import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import joblib

from .feature_extraction import load_dataset_from_ravdess_tess
from src.audio_module.model_audio import build_audio_lstm

# Configuration (adjust paths as needed)
RAVDESS_PATH = "./dataset/RAVDESS/"
TESS_PATH = "./dataset/TESS/"
MODEL_DIR = "./model"
MODEL_PATH = os.path.join(MODEL_DIR, "audio_lstm_model.h5")
ENCODER_PATH = os.path.join(MODEL_DIR, "audio_label_encoder.pkl")

# Dataset emotion code mapping for RAVDESS
RAVDESS_EMOTIONS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}
TESS_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant', 'sad', 'surprise']

def map_to_mental_health(emotion):
    """
    Map raw emotion labels to mental-health categories used in this project.
    Modify mapping logic as needed.
    """
    if emotion is None:
        return None
    emotion = emotion.lower()
    if emotion in ['happy', 'calm', 'pleasant', 'neutral']:
        return 'normal'
    elif emotion in ['sad', 'fearful', 'disgust']:
        return 'depressed'
    elif emotion in ['angry', 'surprised']:
        return 'stressed'
    else:
        return None

def prepare_data(max_len=216, n_mfcc=40):
    print("Loading and extracting features from datasets...")
    X_list, y_list = load_dataset_from_ravdess_tess(
        RAVDESS_PATH, TESS_PATH, RAVDESS_EMOTIONS, TESS_EMOTIONS, map_to_mental_health,
        max_len=max_len, n_mfcc=n_mfcc
    )

    if len(X_list) == 0:
        raise ValueError("No data found. Check dataset paths and file structure.")

    X = np.stack(X_list)  # shape (N, timesteps, features)
    y = np.array(y_list)

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_cat = to_categorical(y_enc)

    # Save label encoder directory (create model dir if missing)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(le, ENCODER_PATH)
    print(f"Saved label encoder to {ENCODER_PATH}")

    return X, y_cat, le

def train(max_len=216, n_mfcc=40, epochs=30, batch_size=32, test_size=0.2):
    X, y_cat, le = prepare_data(max_len=max_len, n_mfcc=n_mfcc)
    input_shape = (X.shape[1], X.shape[2])  # (timesteps, features)
    num_classes = y_cat.shape[1]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=test_size, random_state=42, stratify=np.argmax(y_cat, axis=1))

    print("Building model...")
    model = build_audio_lstm(input_shape=input_shape, num_classes=num_classes)

    # Callbacks
    os.makedirs(MODEL_DIR, exist_ok=True)
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early]
    )

    # Save final model (if not already saved by checkpoint)
    if not os.path.exists(MODEL_PATH):
        model.save(MODEL_PATH)
    print(f"Model training complete. Best model saved to {MODEL_PATH}")

    return model, history, le

if __name__ == "__main__":
    # Example run (adjust hyperparameters if needed)
    model, history, le = train(max_len=216, n_mfcc=40, epochs=30, batch_size=32)
