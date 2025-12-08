# image_module/src/preprocessing.py
"""
Preprocessing script for facial emotion recognition dataset.
Processes FER-2013 or similar datasets and maps emotions to mental health states.
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
from image_utils import (
    load_and_preprocess_image,
    map_emotion_to_mental_health
)

# Configuration
FER2013_PATH = "./data/raw/FER2013"
OUTPUT_PATH = "./data/processed/image_mental_health_features.npz"
TARGET_SIZE = (48, 48)  # FER-2013 native size
GRAYSCALE = True

# Emotion labels in FER-2013
FER2013_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def load_fer2013_dataset(data_path, subset='train'):
    """
    Load FER-2013 dataset from directory structure.
    
    Args:
        data_path (str or Path): Path to FER-2013 root directory
        subset (str): 'train' or 'test'
        
    Returns:
        tuple: (images, labels, emotions)
    """
    data_path = Path(data_path) / subset
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")
    
    images = []
    labels = []
    emotions = []
    
    print(f"\n📂 Loading {subset} data from: {data_path}")
    
    # Iterate through emotion folders
    for emotion_folder in sorted(data_path.iterdir()):
        if not emotion_folder.is_dir():
            continue
        
        emotion = emotion_folder.name.lower()
        
        if emotion not in FER2013_EMOTIONS:
            print(f"⚠️  Skipping unknown emotion folder: {emotion}")
            continue
        
        # Map to mental health state
        mental_state = map_emotion_to_mental_health(emotion)
        
        if mental_state is None:
            print(f"⚠️  Could not map emotion '{emotion}' to mental health state")
            continue
        
        # Load images from this emotion folder
        image_files = list(emotion_folder.glob('*.jpg')) + \
                     list(emotion_folder.glob('*.png'))
        
        print(f"   {emotion:10s} → {mental_state:10s}: {len(image_files)} images")
        
        for img_path in tqdm(image_files, desc=f"   Processing {emotion}", leave=False):
            # Load and preprocess image
            img = load_and_preprocess_image(
                img_path,
                target_size=TARGET_SIZE,
                detect_face=False,  # FER-2013 already has cropped faces
                grayscale=GRAYSCALE
            )
            
            if img is not None:
                images.append(img)
                labels.append(mental_state)
                emotions.append(emotion)
    
    return np.array(images), np.array(labels), np.array(emotions)


def preprocess_fer2013():
    """
    Main preprocessing pipeline for FER-2013 dataset.
    """
    print("=" * 60)
    print("IMAGE MODULE - PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load training data
    print("\n📥 Loading training data...")
    X_train, y_train, emotions_train = load_fer2013_dataset(FER2013_PATH, 'train')
    print(f"   Loaded {len(X_train)} training samples")
    
    # Step 2: Load test data
    print("\n📥 Loading test data...")
    X_test, y_test, emotions_test = load_fer2013_dataset(FER2013_PATH, 'test')
    print(f"   Loaded {len(X_test)} test samples")
    
    # Step 3: Display class distribution
    print("\n📊 Training set class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"   {label:10s}: {count:5d} samples ({count/len(y_train)*100:.1f}%)")
    
    print("\n📊 Test set class distribution:")
    unique, counts = np.unique(y_test, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"   {label:10s}: {count:5d} samples ({count/len(y_test)*100:.1f}%)")
    
    # Step 4: Reshape for CNN input
    if GRAYSCALE:
        # Add channel dimension for grayscale
        X_train = X_train.reshape(-1, TARGET_SIZE[0], TARGET_SIZE[1], 1)
        X_test = X_test.reshape(-1, TARGET_SIZE[0], TARGET_SIZE[1], 1)
    else:
        # Ensure RGB has 3 channels
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(-1, TARGET_SIZE[0], TARGET_SIZE[1], 1)
            X_test = X_test.reshape(-1, TARGET_SIZE[0], TARGET_SIZE[1], 1)
    
    print(f"\n📐 Data shapes:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   y_train: {y_train.shape}")
    print(f"   y_test: {y_test.shape}")
    
    # Step 5: Save processed data
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        emotions_train=emotions_train,
        emotions_test=emotions_test
    )
    
    print(f"\n💾 Saved preprocessed data to: {output_path}")
    
    # Step 6: Summary
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total training samples: {len(X_train)}")
    print(f"Total test samples: {len(X_test)}")
    print(f"Image size: {TARGET_SIZE}")
    print(f"Color mode: {'Grayscale' if GRAYSCALE else 'RGB'}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"Classes: {sorted(np.unique(y_train))}")
    print(f"\nOutput file: {output_path}")
    print("=" * 60)
    print("✅ Preprocessing complete!")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test = preprocess_fer2013()
    except FileNotFoundError as e:
        print(f"\n❌ Error: Dataset not found!")
        print(f"   Please download FER-2013 dataset and place it in:")
        print(f"   {FER2013_PATH}/")
        print(f"   See README.md for download instructions.")
        print(f"\n   {e}")
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        raise
