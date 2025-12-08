# text_module/src/text_utils.py
"""
Utility functions for text preprocessing and data handling.
"""

import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
from pathlib import Path

# Download stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


def clean_text(text):
    """
    Clean and preprocess text data.
    
    Args:
        text (str): Raw text input
        
    Returns:
        str: Cleaned text
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove stopwords
    sw = set(stopwords.words('english'))
    words = text.split()
    text = ' '.join([w for w in words if w not in sw])
    
    return text


def load_dataset(file_path):
    """
    Load and perform initial cleaning of the dataset.
    
    Args:
        file_path (str or Path): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded and cleaned dataframe
    """
    # Load data
    data = pd.read_csv(file_path).dropna()
    
    # Remove unnamed columns
    unnamed_cols = [c for c in data.columns if c.startswith('Unnamed')]
    if unnamed_cols:
        data = data.drop(columns=unnamed_cols)
    
    return data


def get_class_distribution(data, label_column='status'):
    """
    Get the distribution of classes in the dataset.
    
    Args:
        data (pd.DataFrame): Dataset
        label_column (str): Name of the label column
        
    Returns:
        pd.Series: Class distribution
    """
    return data[label_column].value_counts()


def save_processed_data(data, output_path):
    """
    Save processed data to pickle file.
    
    Args:
        data (pd.DataFrame): Processed dataframe
        output_path (str or Path): Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_pickle(output_path)
    print(f"✅ Saved processed data to: {output_path}")


def load_processed_data(file_path):
    """
    Load processed data from pickle file.
    
    Args:
        file_path (str or Path): Path to pickle file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    return pd.read_pickle(file_path)
