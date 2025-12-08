# image_module/src/evaluate.py
"""
Evaluation script for trained CNN facial emotion recognition model.
Generates classification reports and visualizations.
"""

import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model

# Configuration
DATA_PATH = "./data/processed/image_mental_health_features.npz"
MODEL_PATH = "./model/image_cnn_model.h5"
ENCODER_PATH = "./model/image_label_encoder.pkl"


def evaluate_model():
    """
    Main evaluation pipeline for CNN model.
    """
    print("=" * 60)
    print("IMAGE MODULE - MODEL EVALUATION")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n📥 Loading preprocessed data...")
    data = np.load(DATA_PATH, allow_pickle=True)
    
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"   Test samples: {len(X_test)}")
    print(f"   Image shape: {X_test.shape[1:]}")
    
    # Step 2: Load label encoder
    print("\n📥 Loading label encoder...")
    label_encoder = joblib.load(ENCODER_PATH)
    print(f"   Classes: {list(label_encoder.classes_)}")
    
    # Encode test labels
    y_test_enc = label_encoder.transform(y_test)
    
    # Step 3: Load model
    print(f"\n🤖 Loading trained model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    
    print("\n📋 Model Summary:")
    model.summary()
    
    # Step 4: Make predictions
    print("\n🔍 Making predictions on test set...")
    y_pred_probs = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Step 5: Calculate metrics
    print("\n📈 Calculating metrics...")
    accuracy = accuracy_score(y_test_enc, y_pred)
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(
        y_test_enc,
        y_pred,
        target_names=label_encoder.classes_,
        digits=4
    ))
    
    # Step 6: Confusion matrix
    print("\n📊 Generating confusion matrix...")
    cm = confusion_matrix(y_test_enc, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - Image Mental Health Classification', fontsize=14, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save confusion matrix
    output_path = Path("./evaluation_results")
    output_path.mkdir(exist_ok=True)
    cm_path = output_path / "confusion_matrix_image.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"   Saved confusion matrix to: {cm_path}")
    
    plt.show()
    
    # Step 7: Per-class accuracy
    print("\n📊 Per-class Accuracy:")
    print("-" * 60)
    for i, class_name in enumerate(label_encoder.classes_):
        class_mask = y_test_enc == i
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == y_test_enc[class_mask]).mean()
            class_count = class_mask.sum()
            print(f"   {class_name:10s}: {class_acc:.4f} ({class_count} samples)")
    
    # Step 8: Confidence distribution
    print("\n📊 Prediction Confidence Distribution:")
    max_probs = np.max(y_pred_probs, axis=1)
    print(f"   Mean confidence: {max_probs.mean():.4f}")
    print(f"   Median confidence: {np.median(max_probs):.4f}")
    print(f"   Min confidence: {max_probs.min():.4f}")
    print(f"   Max confidence: {max_probs.max():.4f}")
    
    # Plot confidence distribution
    plt.figure(figsize=(10, 6))
    plt.hist(max_probs, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Prediction Confidence', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Prediction Confidence', fontsize=14)
    plt.axvline(max_probs.mean(), color='red', linestyle='--', 
                label=f'Mean: {max_probs.mean():.3f}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save confidence distribution
    conf_path = output_path / "confidence_distribution.png"
    plt.savefig(conf_path, dpi=300, bbox_inches='tight')
    print(f"   Saved confidence distribution to: {conf_path}")
    
    plt.show()
    
    # Step 9: Sample predictions visualization
    print("\n📸 Visualizing sample predictions...")
    visualize_sample_predictions(X_test, y_test, y_pred, y_pred_probs, 
                                 label_encoder, output_path)
    
    # Step 10: Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total test samples: {len(X_test)}")
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Mean prediction confidence: {max_probs.mean():.4f}")
    print(f"\nResults saved to: {output_path}")
    print("=" * 60)
    print("✅ Evaluation complete!")
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'true_labels': y_test_enc,
        'confusion_matrix': cm,
        'confidence': max_probs
    }


def visualize_sample_predictions(X_test, y_test, y_pred, y_pred_probs, 
                                label_encoder, output_path, num_samples=16):
    """
    Visualize sample predictions with images.
    
    Args:
        X_test: Test images
        y_test: True labels
        y_pred: Predicted labels
        y_pred_probs: Prediction probabilities
        label_encoder: Label encoder
        output_path: Path to save visualization
        num_samples: Number of samples to visualize
    """
    # Encode true labels
    y_test_enc = label_encoder.transform(y_test)
    
    # Select random samples
    indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
    
    # Create figure
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx, ax in enumerate(axes):
        if idx >= len(indices):
            ax.axis('off')
            continue
        
        i = indices[idx]
        
        # Get image
        img = X_test[i]
        if img.shape[-1] == 1:
            img = img.squeeze()
        
        # Display image
        ax.imshow(img, cmap='gray')
        
        # Get labels and confidence
        true_label = label_encoder.classes_[y_test_enc[i]]
        pred_label = label_encoder.classes_[y_pred[i]]
        confidence = y_pred_probs[i][y_pred[i]]
        
        # Color based on correctness
        color = 'green' if y_test_enc[i] == y_pred[i] else 'red'
        
        # Set title
        title = f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}"
        ax.set_title(title, fontsize=10, color=color, weight='bold')
        ax.axis('off')
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', 
                fontsize=14, weight='bold')
    plt.tight_layout()
    
    # Save
    sample_path = output_path / "sample_predictions.png"
    plt.savefig(sample_path, dpi=300, bbox_inches='tight')
    print(f"   Saved sample predictions to: {sample_path}")
    
    plt.show()


if __name__ == "__main__":
    try:
        results = evaluate_model()
    except FileNotFoundError as e:
        print(f"\n❌ Error: Required file not found!")
        print(f"   Please ensure model is trained first.")
        print(f"   {e}")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        raise
