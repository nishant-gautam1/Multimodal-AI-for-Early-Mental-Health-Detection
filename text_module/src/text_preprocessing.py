# text_module/src/text_preprocessing.py
"""
Preprocessing script for text-based mental health classification.
Converts raw text data into a format suitable for BERT model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import pickle
from text_utils import clean_text, load_dataset, get_class_distribution, save_processed_data

# Configuration
DATA_PATH = "./data/Combined_Data_Expanded.csv"
OUTPUT_DIR = "./data/processed/"
OUTPUT_FILE = "text_mental_health_processed.pkl"
LABEL_ENCODER_PATH = "./model/label_encoder.pkl"

# Sample size (set to None to use full dataset)
SAMPLE_SIZE = 6000
RANDOM_STATE = 42


def preprocess_text_data():
    """
    Main preprocessing pipeline for text data.
    """
    print("=" * 60)
    print("TEXT MODULE - PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load dataset
    print("\n📥 Loading dataset...")
    data = load_dataset(DATA_PATH)
    print(f"   Loaded {len(data)} samples")
    print(f"   Columns: {list(data.columns)}")
    
    # Step 2: Sample data if needed
    if SAMPLE_SIZE and SAMPLE_SIZE < len(data):
        print(f"\n🎲 Sampling {SAMPLE_SIZE} records...")
        data = data.sample(min(SAMPLE_SIZE, len(data)), random_state=RANDOM_STATE).reset_index(drop=True)
    
    # Step 3: Display initial class distribution
    print("\n📊 Initial class distribution:")
    print(get_class_distribution(data))
    
    # Step 4: Clean text
    print("\n🧹 Cleaning text data...")
    data['statement'] = data['statement'].apply(clean_text)
    print("   Text cleaning complete")
    
    # Step 5: Balance classes using oversampling
    print("\n⚖️ Balancing classes using RandomOverSampler...")
    X = data.drop('status', axis=1)
    y = data['status']
    
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    
    # Combine back into dataframe
    data = pd.concat([X_resampled, y_resampled], axis=1)
    
    print("\n📊 Balanced class distribution:")
    print(get_class_distribution(data))
    print(f"   Total samples after balancing: {len(data)}")
    
    # Step 6: Encode labels
    print("\n🔤 Encoding labels...")
    le = LabelEncoder()
    data['label'] = le.fit_transform(data['status'])
    
    print(f"   Classes: {list(le.classes_)}")
    print(f"   Label mapping:")
    for idx, class_name in enumerate(le.classes_):
        print(f"      {idx}: {class_name}")
    
    # Step 7: Save label encoder
    Path(LABEL_ENCODER_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)
    print(f"\n💾 Saved label encoder to: {LABEL_ENCODER_PATH}")
    
    # Step 8: Save processed data
    output_path = Path(OUTPUT_DIR) / OUTPUT_FILE
    save_processed_data(data, output_path)
    
    # Step 9: Display summary
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(data)}")
    print(f"Number of classes: {len(le.classes_)}")
    print(f"Features: statement (cleaned text)")
    print(f"Labels: status (original), label (encoded)")
    print(f"\nOutput files:")
    print(f"  - Processed data: {output_path}")
    print(f"  - Label encoder: {LABEL_ENCODER_PATH}")
    print("=" * 60)
    print("✅ Preprocessing complete!")
    
    return data, le


if __name__ == "__main__":
    try:
        data, label_encoder = preprocess_text_data()
    except FileNotFoundError as e:
        print(f"\n❌ Error: Dataset file not found!")
        print(f"   Please ensure '{DATA_PATH}' exists.")
        print(f"   {e}")
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        raise
