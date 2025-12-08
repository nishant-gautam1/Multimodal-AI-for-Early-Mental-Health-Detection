# text_module/src/evaluate_text.py
"""
Evaluation script for trained BERT mental health classification model.
Generates classification reports and visualizations.
"""

import pandas as pd
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
from text_utils import load_processed_data

# Configuration
PROCESSED_DATA_PATH = "./data/processed/text_mental_health_processed.pkl"
LABEL_ENCODER_PATH = "./model/label_encoder.pkl"
MODEL_DIR = "./model/saved_mental_status_bert"
MAX_LENGTH = 200


def predict_batch(texts, model, tokenizer, device):
    """
    Predict mental health status for a batch of texts.
    
    Args:
        texts (list): List of text strings
        model: Trained BERT model
        tokenizer: BERT tokenizer
        device: torch device
        
    Returns:
        np.array: Predicted class indices
    """
    # Tokenize
    inputs = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
    
    return predictions.cpu().numpy()


def evaluate_model():
    """
    Main evaluation pipeline for BERT model.
    """
    print("=" * 60)
    print("TEXT MODULE - MODEL EVALUATION")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n📥 Loading preprocessed data...")
    data = load_processed_data(PROCESSED_DATA_PATH)
    print(f"   Loaded {len(data)} samples")
    
    # Step 2: Load label encoder
    print("\n📥 Loading label encoder...")
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    
    print(f"   Classes: {list(label_encoder.classes_)}")
    
    # Step 3: Load model and tokenizer
    print(f"\n🤖 Loading trained model from {MODEL_DIR}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    
    # Step 4: Prepare data
    print("\n📊 Preparing evaluation data...")
    texts = data['statement'].tolist()
    true_labels = data['label'].values
    
    # Step 5: Make predictions
    print("\n🔍 Making predictions...")
    batch_size = 32
    all_predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_preds = predict_batch(batch_texts, model, tokenizer, device)
        all_predictions.extend(batch_preds)
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"   Processed {i + len(batch_texts)}/{len(texts)} samples")
    
    predictions = np.array(all_predictions)
    
    # Step 6: Calculate metrics
    print("\n📈 Calculating metrics...")
    accuracy = accuracy_score(true_labels, predictions)
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(
        true_labels,
        predictions,
        target_names=label_encoder.classes_,
        digits=4
    ))
    
    # Step 7: Confusion matrix
    print("\n📊 Generating confusion matrix...")
    cm = confusion_matrix(true_labels, predictions)
    
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
    plt.title('Confusion Matrix - Text Mental Health Classification', fontsize=14, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save confusion matrix
    output_path = Path("./evaluation_results")
    output_path.mkdir(exist_ok=True)
    cm_path = output_path / "confusion_matrix_text.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"   Saved confusion matrix to: {cm_path}")
    
    plt.show()
    
    # Step 8: Per-class accuracy
    print("\n📊 Per-class Accuracy:")
    print("-" * 60)
    for i, class_name in enumerate(label_encoder.classes_):
        class_mask = true_labels == i
        class_acc = (predictions[class_mask] == true_labels[class_mask]).mean()
        class_count = class_mask.sum()
        print(f"   {class_name:20s}: {class_acc:.4f} ({class_count} samples)")
    
    # Step 9: Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(data)}")
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print("=" * 60)
    print("✅ Evaluation complete!")
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'true_labels': true_labels,
        'confusion_matrix': cm
    }


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
