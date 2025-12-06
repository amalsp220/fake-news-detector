"""Text preprocessing and dataset preparation for fake news detection."""

import os
import re
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Tuple, Dict, List


class TextPreprocessor:
    """Clean and normalize text data."""
    
    def __init__(self, lowercase=True, remove_special=True, remove_stopwords=False):
        self.lowercase = lowercase
        self.remove_special = remove_special
        self.remove_stopwords = remove_stopwords
    
    def preprocess(self, text: str) -> str:
        """Clean a single text string."""
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters if needed
        if self.remove_special:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text


class FakeNewsDataset:
    """Load and process FakeNewsNet dataset."""
    
    def __init__(self, data_path: str = "data/raw", processed_path: str = "data/processed"):
        self.data_path = data_path
        self.processed_path = processed_path
        self.preprocessor = TextPreprocessor()
    
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Load CSV file."""
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records from {filepath}")
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare dataframe."""
        df = df.copy()
        
        # Combine title and text
        if 'title' in df.columns and 'text' in df.columns:
            df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        elif 'text' in df.columns:
            df['full_text'] = df['text']
        elif 'title' in df.columns:
            df['full_text'] = df['title']
        else:
            raise ValueError("No text columns found")
        
        # Clean text
        df['full_text'] = df['full_text'].apply(self.preprocessor.preprocess)
        
        # Remove empty texts
        df = df[df['full_text'].str.len() > 10].copy()
        
        # Ensure label column
        if 'label' not in df.columns and 'fake' in df.columns:
            df['label'] = df['fake']
        
        return df
    
    def create_splits(self, df: pd.DataFrame, 
                     train_size: float = 0.7,
                     val_size: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/val/test with stratification."""
        test_size = 1 - train_size - val_size
        
        # First split: train and temp
        train, temp = train_test_split(
            df, test_size=(val_size + test_size),
            stratify=df['label'],
            random_state=42
        )
        
        # Second split: val and test
        val_ratio = val_size / (val_size + test_size)
        val, test = train_test_split(
            temp, test_size=(1 - val_ratio),
            stratify=temp['label'],
            random_state=42
        )
        
        return train, val, test


def process_dataset(raw_path: str = "data/raw", output_path: str = "data/processed") -> None:
    """Main function to process dataset."""
    os.makedirs(output_path, exist_ok=True)
    
    print("Loading raw data...")
    dataset = FakeNewsDataset(raw_path, output_path)
    
    # Load all CSV files in raw directory
    all_dfs = []
    for file in os.listdir(raw_path):
        if file.endswith(".csv"):
            filepath = os.path.join(raw_path, file)
            try:
                df = dataset.load_csv(filepath)
                df = dataset.prepare_data(df)
                all_dfs.append(df)
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    if not all_dfs:
        print("No CSV files found. Creating mock dataset for demo...")
        # Create mock dataset for demo
        all_dfs = [pd.DataFrame({
            'full_text': ['this is fake news'] * 100 + ['this is real news'] * 100,
            'label': [1] * 100 + [0] * 100,
            'source': ['demo'] * 200
        })]
    
    # Combine all data
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined dataset size: {len(combined_df)}")
    
    # Create splits
    print("Creating train/val/test splits...")
    train_df, val_df, test_df = dataset.create_splits(combined_df)
    
    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    
    # Save splits
    train_df.to_csv(os.path.join(output_path, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_path, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_path, 'test.csv'), index=False)
    
    print(f"Data saved to {output_path}")


if __name__ == "__main__":
    process_dataset()
