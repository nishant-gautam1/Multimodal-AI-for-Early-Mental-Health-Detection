# image_module/flask_app/app_image.py
"""
Flask API for image-based mental health detection.
Provides web interface and API endpoints for facial emotion recognition.
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from image_utils import load_and_preprocess_image, detect_face_haar, crop_face, preprocess_image

# Configuration
MODEL_PATH = "../model/image_cnn_model.h5"
ENCODER_PATH = "../model/image_label_encoder.pkl"
TARGET_SIZE = (48, 48)
GRAYSCALE = True

# Flask app
app = Flask(__name__, template_folder="templates")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load model & encoder once at startup
print("=" * 60)
print("IMAGE MODULE - FLASK API")
print("=" * 60)
print("\n📥 Loading model and encoder...")

try:
    model_path = Path(__file__).parent / MODEL_PATH
    encoder_path = Path(__file__).parent / ENCODER_PATH
    
    model = load_model(str(model_path))
    label_encoder = joblib.load(str(encoder_path))
    
    print(f"✅ Model loaded from: {model_path}")
    print(f"✅ Encoder loaded from: {encoder_path}")
    print(f"   Classes: {list(label_encoder.classes_)}")
    print("=" * 60)
    
except Exception as e:
    print(f"❌ Error loading model/encoder: {e}")
    print("   Please ensure model is trained first.")
    model = None
    label_encoder = None


def extract_features_for_inference(image_bytes):
    """
    Extract features from uploaded image for inference.
    
    Args:
        image_bytes: Image file bytes
        
    Returns:
        np.array: Preprocessed image ready for model, or None if error
    """
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return None, "Could not decode image"
        
        # Detect face
        face_coords = detect_face_haar(image)
        
        if face_coords is None:
            # Try without face detection (use whole image)
            print("⚠️  No face detected, using whole image")
            processed = preprocess_image(image, TARGET_SIZE, GRAYSCALE)
        else:
            # Crop face and preprocess
            face = crop_face(image, face_coords)
            processed = preprocess_image(face, TARGET_SIZE, GRAYSCALE)
        
        # Reshape for model input
        if GRAYSCALE:
            processed = processed.reshape(1, TARGET_SIZE[0], TARGET_SIZE[1], 1)
        else:
            processed = processed.reshape(1, TARGET_SIZE[0], TARGET_SIZE[1], 3)
        
        return processed, None
        
    except Exception as e:
        return None, f"Error processing image: {str(e)}"


# Routes
@app.route("/", methods=["GET"])
def index():
    """Simple upload page."""
    return render_template("image_upload.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Form POST (from web UI): expects 'file' (image)
    Returns rendered page with prediction.
    """
    if model is None or label_encoder is None:
        return render_template("image_upload.html", 
                             error="Model not loaded. Please train the model first.")
    
    if "file" not in request.files:
        return render_template("image_upload.html", 
                             error="No file uploaded. Please upload an image file.")
    
    file = request.files["file"]
    
    if file.filename == "":
        return render_template("image_upload.html", 
                             error="Empty filename. Please upload a valid image file.")
    
    # Check file extension
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        return render_template("image_upload.html",
                             error=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}")
    
    # Read image bytes
    image_bytes = file.read()
    
    # Extract features
    features, error = extract_features_for_inference(image_bytes)
    
    if features is None:
        return render_template("image_upload.html", error=error)
    
    # Predict
    try:
        predictions = model.predict(features, verbose=0)
        class_index = int(np.argmax(predictions, axis=1)[0])
        label = label_encoder.inverse_transform([class_index])[0]
        confidence = float(np.max(predictions))
        
        return render_template("image_upload.html",
                             prediction=label,
                             confidence=f"{confidence*100:.2f}%")
    
    except Exception as e:
        return render_template("image_upload.html",
                             error=f"Prediction error: {str(e)}")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    API prediction endpoint (programmatic).
    Accepts multipart form 'file' (image) and returns JSON:
    { "label": "...", "confidence": 0.92 }
    """
    if model is None or label_encoder is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use key 'file' with multipart/form-data."}), 400
    
    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    
    # Read image bytes
    image_bytes = file.read()
    
    # Extract features
    features, error = extract_features_for_inference(image_bytes)
    
    if features is None:
        return jsonify({"error": error}), 400
    
    # Predict
    try:
        predictions = model.predict(features, verbose=0)
        class_index = int(np.argmax(predictions, axis=1)[0])
        label = label_encoder.inverse_transform([class_index])[0]
        confidence = float(np.max(predictions))
        
        # Get all class probabilities
        probs = {
            label_encoder.inverse_transform([i])[0]: float(predictions[0][i])
            for i in range(len(label_encoder.classes_))
        }
        
        return jsonify({
            "label": label,
            "confidence": confidence,
            "probabilities": probs
        })
    
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "encoder_loaded": label_encoder is not None,
        "classes": list(label_encoder.classes_) if label_encoder else None
    })


@app.route("/api/info", methods=["GET"])
def api_info():
    """API information endpoint."""
    return jsonify({
        "module": "image",
        "model_type": "CNN",
        "input_size": TARGET_SIZE,
        "color_mode": "grayscale" if GRAYSCALE else "rgb",
        "classes": list(label_encoder.classes_) if label_encoder else None,
        "endpoints": [
            {"path": "/", "method": "GET", "description": "Web interface"},
            {"path": "/predict", "method": "POST", "description": "Form-based prediction"},
            {"path": "/api/predict", "method": "POST", "description": "JSON API prediction"},
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/api/info", "method": "GET", "description": "API information"}
        ]
    })


# Run
if __name__ == "__main__":
    # Use port 5002 to avoid conflict with audio (5000) and text (5000) modules
    print("\n🚀 Starting Flask server...")
    print("   Access the web interface at: http://localhost:5002")
    print("   API endpoint: http://localhost:5002/api/predict")
    print("\n   Press Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(debug=True, host="0.0.0.0", port=5002)
