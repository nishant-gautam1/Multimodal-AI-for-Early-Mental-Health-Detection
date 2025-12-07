# flask_app/app_audio_lstm.py
import os
from flask import Flask, render_template, request, jsonify, send_file
import joblib
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from src.audio_utils import extract_mfcc   # adjust import if your project layout differs


# Configuration
MODEL_PATH = "./model/audio_lstm_model.h5"
ENCODER_PATH = "./model/audio_label_encoder.pkl"
# Reference to your processed dataset (uploaded / used earlier)
DATASET_FILE = "/data/audio_mental_health_features.npz"

# Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

# Load model & encoder once at app startup
print("Loading model and encoder...")
model = load_model(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
print("Model and encoder loaded successfully.")

# Parameters must match training
MAX_LEN = 216
N_MFCC = 40


# feature extraction 
def extract_features_for_inference(file_path):
    """
    Extract MFCC features for a single audio file and return
    a numpy array shaped (1, n_mfcc, max_len) suitable for LSTM model.
    """
    mfcc = extract_mfcc(file_path, n_mfcc=N_MFCC, max_len=MAX_LEN)
    if mfcc is None:
        return None
    # expand dims -> (1, n_mfcc, max_len)
    return np.expand_dims(mfcc, axis=0).astype(np.float32)


# Routes
@app.route("/", methods=["GET"])
def index():
    """Simple upload page."""
    return render_template("audio_upload.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Form POST (from web UI): expects 'file' (wav)
    Returns rendered page with prediction.
    """
    if "file" not in request.files:
        return render_template("audio_upload.html", error="No file uploaded. Please upload a .wav file.")

    file = request.files["file"]
    if file.filename == "":
        return render_template("audio_upload.html", error="Empty filename. Please upload a valid .wav file.")

    # save temporarily
    temp_path = "./temp_audio.wav"
    file.save(temp_path)

    # extract features
    features = extract_features_for_inference(temp_path)
    if features is None:
        os.remove(temp_path)
        return render_template("audio_upload.html", error="Could not extract features from the audio. Try a different file.")

    # predict
    preds = model.predict(features)  # shape (1, num_classes)
    class_index = int(np.argmax(preds, axis=1)[0])
    label = label_encoder.inverse_transform([class_index])[0]
    confidence = float(np.max(preds))

    # cleanup
    os.remove(temp_path)

    # render
    return render_template("audio_upload.html",
                           prediction=label,
                           confidence=f"{confidence*100:.2f}%")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    API prediction endpoint (programmatic).
    Accepts multipart form 'file' (wav) and returns JSON:
    { "label": "...", "confidence": 0.92 }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use key 'file' with multipart/form-data."}), 400

    file = request.files["file"]
    temp_path = "./temp_audio.wav"
    file.save(temp_path)

    features = extract_features_for_inference(temp_path)
    if features is None:
        os.remove(temp_path)
        return jsonify({"error": "Could not extract features from audio."}), 400

    preds = model.predict(features)
    class_index = int(np.argmax(preds, axis=1)[0])
    label = label_encoder.inverse_transform([class_index])[0]
    confidence = float(np.max(preds))

    os.remove(temp_path)
    return jsonify({"label": label, "confidence": confidence})

@app.route("/dataset", methods=["GET"])
def download_dataset():
    """
    Optional: serve the processed dataset (for debugging / report),
    using the uploaded local file path you provided earlier.
    """
    if not os.path.exists(DATASET_FILE):
        return jsonify({"error": f"Dataset file not found at {DATASET_FILE}"}), 404
    # send file directly (binary)
    return send_file(DATASET_FILE, as_attachment=True)

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    # set host=0.0.0.0 if you want to expose on local network
    app.run(debug=True, port=5000)
