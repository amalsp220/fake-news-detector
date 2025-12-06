#!/usr/bin/env python
"""Training script for baseline models."""

import os
import pandas as pd
from models_baseline import train_baseline
from preprocess import process_dataset


def main():
    """Main training script."""
    # Prepare data
    print("Preparing data...")
    process_dataset()
    
    # Load data
    print("Loading processed data...")
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    
    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    
    # Train models
    models = ['logistic', 'svm', 'rf']
    
    for model_type in models:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()} model")
        print(f"{'='*50}")
        
        model, metrics = train_baseline(
            train_df['full_text'],
            val_df['full_text'],
            train_df['label'].values,
            val_df['label'].values,
            model_type=model_type
        )
        
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        
        # Save model
        os.makedirs("models", exist_ok=True)
        model.save(f"models/baseline_{model_type}")
        print(f"Model saved to models/baseline_{model_type}")


if __name__ == "__main__":
    main()
