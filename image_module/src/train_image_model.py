# image_module/src/train_image_model.py
"""
Training script for CNN-based facial emotion recognition model.
Trains a convolutional neural network for mental health state classification from facial images.
"""

import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, Activation
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Configuration
DATA_PATH = "./data/processed/image_mental_health_features.npz"
MODEL_PATH = "./model/image_cnn_model.h5"
ENCODER_PATH = "./model/image_label_encoder.pkl"

# Model parameters
INPUT_SHAPE = (48, 48, 1)  # Grayscale images
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Data augmentation parameters
AUGMENTATION = True
ROTATION_RANGE = 20
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
HORIZONTAL_FLIP = True
ZOOM_RANGE = 0.1


def build_cnn_model(input_shape, num_classes):
    """
    Build CNN model for facial emotion recognition.
    
    Args:
        input_shape (tuple): Input image shape (height, width, channels)
        num_classes (int): Number of output classes
        
    Returns:
        keras.Model: Compiled CNN model
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 2
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 3
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 4
        Conv2D(256, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Fully connected layers
        Flatten(),
        Dense(512),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        
        Dense(256),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    return model


def train_model():
    """
    Main training pipeline for CNN model.
    """
    print("=" * 60)
    print("IMAGE MODULE - CNN MODEL TRAINING")
    print("=" * 60)
    
    # Step 1: Load preprocessed data
    print("\n📥 Loading preprocessed data...")
    data = np.load(DATA_PATH, allow_pickle=True)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Image shape: {X_train.shape[1:]}")
    
    # Step 2: Encode labels
    print("\n🔤 Encoding labels...")
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    # Convert to categorical
    num_classes = len(le.classes_)
    y_train_cat = to_categorical(y_train_enc, num_classes)
    y_test_cat = to_categorical(y_test_enc, num_classes)
    
    print(f"   Number of classes: {num_classes}")
    print(f"   Classes: {list(le.classes_)}")
    
    # Save label encoder
    os.makedirs(Path(ENCODER_PATH).parent, exist_ok=True)
    joblib.dump(le, ENCODER_PATH)
    print(f"   Saved label encoder to: {ENCODER_PATH}")
    
    # Step 3: Build model
    print("\n🤖 Building CNN model...")
    model = build_cnn_model(X_train.shape[1:], num_classes)
    
    # Compile model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    print("\n📋 Model Summary:")
    model.summary()
    
    # Step 4: Set up data augmentation
    if AUGMENTATION:
        print("\n🔄 Setting up data augmentation...")
        train_datagen = ImageDataGenerator(
            rotation_range=ROTATION_RANGE,
            width_shift_range=WIDTH_SHIFT_RANGE,
            height_shift_range=HEIGHT_SHIFT_RANGE,
            horizontal_flip=HORIZONTAL_FLIP,
            zoom_range=ZOOM_RANGE,
            validation_split=VALIDATION_SPLIT
        )
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator(validation_split=VALIDATION_SPLIT)
        
        # Create generators
        train_generator = train_datagen.flow(
            X_train, y_train_cat,
            batch_size=BATCH_SIZE,
            subset='training'
        )
        
        val_generator = val_datagen.flow(
            X_train, y_train_cat,
            batch_size=BATCH_SIZE,
            subset='validation'
        )
        
        print(f"   Augmentation enabled:")
        print(f"      Rotation: ±{ROTATION_RANGE}°")
        print(f"      Width shift: ±{WIDTH_SHIFT_RANGE*100}%")
        print(f"      Height shift: ±{HEIGHT_SHIFT_RANGE*100}%")
        print(f"      Horizontal flip: {HORIZONTAL_FLIP}")
        print(f"      Zoom: ±{ZOOM_RANGE*100}%")
    
    # Step 5: Set up callbacks
    print("\n⚙️ Setting up training callbacks...")
    
    callbacks = [
        # Save best model
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Step 6: Train model
    print("\n🚀 Starting training...")
    print("=" * 60)
    
    if AUGMENTATION:
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            X_train, y_train_cat,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=VALIDATION_SPLIT,
            callbacks=callbacks,
            verbose=1
        )
    
    # Step 7: Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    
    print(f"\n   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    
    # Step 8: Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Model architecture: Custom CNN")
    print(f"Input shape: {X_train.shape[1:]}")
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Epochs trained: {len(history.history['loss'])}")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Data augmentation: {'Enabled' if AUGMENTATION else 'Disabled'}")
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Label encoder saved to: {ENCODER_PATH}")
    print("=" * 60)
    print("✅ Training complete!")
    
    return model, history


if __name__ == "__main__":
    try:
        import tensorflow as tf
        
        # Check for GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"\n🖥️  GPU available: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"   {gpu}")
        else:
            print("\n🖥️  Running on CPU")
        
        model, history = train_model()
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Preprocessed data not found!")
        print(f"   Please run preprocessing.py first.")
        print(f"   {e}")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise
