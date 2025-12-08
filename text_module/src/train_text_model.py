# text_module/src/train_text_model.py
"""
Training script for BERT-based mental health classification model.
Fine-tunes BERT for sequence classification on preprocessed text data.
"""

import pandas as pd
import numpy as np
import pickle
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from text_utils import load_processed_data

# Configuration
PROCESSED_DATA_PATH = "./data/processed/text_mental_health_processed.pkl"
LABEL_ENCODER_PATH = "./model/label_encoder.pkl"
MODEL_OUTPUT_DIR = "./model/saved_mental_status_bert"

# Model parameters
BERT_MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 200
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
TEST_SIZE = 0.2
RANDOM_STATE = 42


def tokenize_function(examples, tokenizer):
    """
    Tokenize text examples for BERT.
    
    Args:
        examples: Dataset examples
        tokenizer: BERT tokenizer
        
    Returns:
        Tokenized examples
    """
    return tokenizer(
        examples['statement'],
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH
    )


def prepare_dataset(data, tokenizer):
    """
    Prepare dataset for training.
    
    Args:
        data (pd.DataFrame): Preprocessed data
        tokenizer: BERT tokenizer
        
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    print("\n📊 Preparing dataset...")
    
    # Split data
    train_df, test_df = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data['label']
    )
    
    print(f"   Train samples: {len(train_df)}")
    print(f"   Test samples: {len(test_df)}")
    
    # Convert to HuggingFace Dataset format
    train_dataset = Dataset.from_pandas(train_df[['statement', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['statement', 'label']])
    
    # Tokenize
    print("\n🔤 Tokenizing text...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )
    
    # Set format for PyTorch
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
    return train_dataset, test_dataset


def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation.
    
    Args:
        eval_pred: Predictions from model
        
    Returns:
        dict: Metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = (predictions == labels).mean()
    
    return {'accuracy': accuracy}


def train_model():
    """
    Main training pipeline for BERT model.
    """
    print("=" * 60)
    print("TEXT MODULE - BERT MODEL TRAINING")
    print("=" * 60)
    
    # Step 1: Load preprocessed data
    print("\n📥 Loading preprocessed data...")
    data = load_processed_data(PROCESSED_DATA_PATH)
    print(f"   Loaded {len(data)} samples")
    
    # Step 2: Load label encoder
    print("\n📥 Loading label encoder...")
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    
    num_labels = len(label_encoder.classes_)
    print(f"   Number of classes: {num_labels}")
    print(f"   Classes: {list(label_encoder.classes_)}")
    
    # Step 3: Initialize tokenizer
    print(f"\n🔧 Loading BERT tokenizer ({BERT_MODEL_NAME})...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    # Step 4: Prepare datasets
    train_dataset, test_dataset = prepare_dataset(data, tokenizer)
    
    # Step 5: Initialize model
    print(f"\n🤖 Initializing BERT model...")
    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=num_labels
    )
    
    # Create label mapping for model config
    id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
    label2id = {label: i for i, label in enumerate(label_encoder.classes_)}
    
    model.config.id2label = id2label
    model.config.label2id = label2id
    
    # Step 6: Set up training arguments
    print("\n⚙️ Configuring training parameters...")
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=f"{MODEL_OUTPUT_DIR}/logs",
        logging_steps=100,
        save_total_limit=2,
        seed=RANDOM_STATE
    )
    
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    
    # Step 7: Initialize trainer
    print("\n🏋️ Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    # Step 8: Train model
    print("\n🚀 Starting training...")
    print("=" * 60)
    trainer.train()
    
    # Step 9: Evaluate model
    print("\n📊 Evaluating model on test set...")
    eval_results = trainer.evaluate()
    print("\nEvaluation Results:")
    for key, value in eval_results.items():
        print(f"   {key}: {value:.4f}")
    
    # Step 10: Save model and tokenizer
    print(f"\n💾 Saving model to {MODEL_OUTPUT_DIR}...")
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    
    # Step 11: Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Model: {BERT_MODEL_NAME}")
    print(f"Number of classes: {num_labels}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Final test accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"\nModel saved to: {MODEL_OUTPUT_DIR}")
    print("=" * 60)
    print("✅ Training complete!")
    
    return model, tokenizer, eval_results


if __name__ == "__main__":
    try:
        # Check for GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n🖥️  Using device: {device}")
        if device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        model, tokenizer, results = train_model()
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Required file not found!")
        print(f"   Please run text_preprocessing.py first.")
        print(f"   {e}")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise
