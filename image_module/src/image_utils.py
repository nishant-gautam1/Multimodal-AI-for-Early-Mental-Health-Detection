# image_module/src/image_utils.py
"""
Utility functions for image processing and face detection.
"""

import cv2
import numpy as np
from pathlib import Path
import os

# Face detection cascade (Haar Cascade)
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'


def detect_face_haar(image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    """
    Detect face in image using Haar Cascade.
    
    Args:
        image (np.array): Input image (grayscale or BGR)
        scale_factor (float): Scale factor for detection
        min_neighbors (int): Minimum neighbors for detection
        min_size (tuple): Minimum face size
        
    Returns:
        tuple: (x, y, w, h) of detected face, or None if no face found
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Load cascade classifier
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size
    )
    
    # Return largest face if multiple detected
    if len(faces) > 0:
        # Sort by area (w * h) and return largest
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        return faces[0]
    
    return None


def detect_face_mtcnn(image):
    """
    Detect face using MTCNN (more accurate but slower).
    
    Args:
        image (np.array): Input image (RGB)
        
    Returns:
        tuple: (x, y, w, h) of detected face, or None if no face found
    """
    try:
        from mtcnn import MTCNN
        
        # Initialize detector
        detector = MTCNN()
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Detect faces
        results = detector.detect_faces(rgb_image)
        
        if len(results) > 0:
            # Get bounding box of first face
            x, y, w, h = results[0]['box']
            return (x, y, w, h)
        
        return None
        
    except ImportError:
        print("⚠️  MTCNN not installed. Using Haar Cascade instead.")
        return detect_face_haar(image)


def crop_face(image, face_coords, padding=0.2):
    """
    Crop face region from image with optional padding.
    
    Args:
        image (np.array): Input image
        face_coords (tuple): (x, y, w, h) face coordinates
        padding (float): Padding ratio around face
        
    Returns:
        np.array: Cropped face image
    """
    x, y, w, h = face_coords
    
    # Add padding
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    
    # Calculate new coordinates with padding
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(image.shape[1], x + w + pad_w)
    y2 = min(image.shape[0], y + h + pad_h)
    
    # Crop face
    face = image[y1:y2, x1:x2]
    
    return face


def preprocess_image(image, target_size=(48, 48), grayscale=True):
    """
    Preprocess image for model input.
    
    Args:
        image (np.array): Input image
        target_size (tuple): Target size (width, height)
        grayscale (bool): Convert to grayscale
        
    Returns:
        np.array: Preprocessed image
    """
    # Convert to grayscale if needed
    if grayscale and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize
    image = cv2.resize(image, target_size)
    
    # Normalize to [0, 1]
    image = image.astype('float32') / 255.0
    
    return image


def load_and_preprocess_image(image_path, target_size=(48, 48), detect_face=True, grayscale=True):
    """
    Load image, detect face, and preprocess for model.
    
    Args:
        image_path (str or Path): Path to image file
        target_size (tuple): Target size for model input
        detect_face (bool): Whether to detect and crop face
        grayscale (bool): Convert to grayscale
        
    Returns:
        np.array: Preprocessed image, or None if face not detected
    """
    # Load image
    image = cv2.imread(str(image_path))
    
    if image is None:
        print(f"⚠️  Could not load image: {image_path}")
        return None
    
    # Detect and crop face if requested
    if detect_face:
        face_coords = detect_face_haar(image)
        
        if face_coords is None:
            print(f"⚠️  No face detected in: {image_path}")
            return None
        
        image = crop_face(image, face_coords)
    
    # Preprocess
    processed = preprocess_image(image, target_size, grayscale)
    
    return processed


def map_emotion_to_mental_health(emotion):
    """
    Map emotion label to mental health state.
    
    Args:
        emotion (str): Emotion label
        
    Returns:
        str: Mental health state (normal, depressed, stressed)
    """
    emotion = emotion.lower()
    
    # Normal state
    if emotion in ['happy', 'neutral', 'surprise']:
        return 'normal'
    
    # Depressed state
    if emotion in ['sad', 'fear', 'disgust']:
        return 'depressed'
    
    # Stressed state
    if emotion in ['angry', 'anger', 'contempt']:
        return 'stressed'
    
    return None


def augment_image(image, rotation_range=20, width_shift=0.1, height_shift=0.1, 
                  horizontal_flip=True, zoom_range=0.1):
    """
    Apply random augmentation to image.
    
    Args:
        image (np.array): Input image
        rotation_range (int): Max rotation degrees
        width_shift (float): Max horizontal shift ratio
        height_shift (float): Max vertical shift ratio
        horizontal_flip (bool): Whether to randomly flip
        zoom_range (float): Max zoom ratio
        
    Returns:
        np.array: Augmented image
    """
    h, w = image.shape[:2]
    
    # Random rotation
    if rotation_range > 0:
        angle = np.random.uniform(-rotation_range, rotation_range)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
    
    # Random shift
    if width_shift > 0 or height_shift > 0:
        tx = np.random.uniform(-width_shift, width_shift) * w
        ty = np.random.uniform(-height_shift, height_shift) * h
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (w, h))
    
    # Random horizontal flip
    if horizontal_flip and np.random.random() > 0.5:
        image = cv2.flip(image, 1)
    
    # Random zoom
    if zoom_range > 0:
        zoom = 1.0 + np.random.uniform(-zoom_range, zoom_range)
        new_h, new_w = int(h * zoom), int(w * zoom)
        image = cv2.resize(image, (new_w, new_h))
        
        # Crop or pad to original size
        if zoom > 1.0:
            # Crop center
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            image = image[start_h:start_h+h, start_w:start_w+w]
        else:
            # Pad
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, 
                                      cv2.BORDER_CONSTANT, value=0)
    
    return image


def visualize_detection(image_path, output_path=None):
    """
    Visualize face detection on image.
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save output (optional)
        
    Returns:
        np.array: Image with face rectangle drawn
    """
    # Load image
    image = cv2.imread(str(image_path))
    
    if image is None:
        print(f"Could not load image: {image_path}")
        return None
    
    # Detect face
    face_coords = detect_face_haar(image)
    
    if face_coords is not None:
        x, y, w, h = face_coords
        # Draw rectangle
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.9, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No face detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(str(output_path), image)
        print(f"Saved visualization to: {output_path}")
    
    return image
