#!/usr/bin/env python
"""Training script for BERT transformer model."""

import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from models_transformer import TransformerNewsClassifier, NewsDataset
from preprocess import process_dataset


def main():
    """Main training script for transformer."""
    # Prepare data
    print("Preparing data...")
    process_dataset()
    
    # Load data
    print("Loading processed data...")
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    
    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerNewsClassifier(device=device)
    
    # Create datasets and loaders
    train_dataset = NewsDataset(
        train_df['full_text'].values,
        train_df['label'].values,
        model.tokenizer
    )
    val_dataset = NewsDataset(
        val_df['full_text'].values,
        val_df['label'].values,
        model.tokenizer
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Setup optimization
    total_steps = len(train_loader) * 3  # 3 epochs
    optimizer = AdamW(model.model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    # Train
    print("Training BERT model...")
    for epoch in range(3):
        print(f"\nEpoch {epoch+1}/3")
        loss = model.train_epoch(train_loader, optimizer, scheduler)
        print(f"Training loss: {loss:.4f}")
        
        metrics = model.evaluate(val_loader)
        print(f"Val Accuracy: {metrics['accuracy']:.4f}")
        print(f"Val F1-Score: {metrics['f1']:.4f}")
    
    # Save model
    model.model.save_pretrained("models/transformer_bert")
    model.tokenizer.save_pretrained("models/transformer_bert")
    print("Model saved to models/transformer_bert")


if __name__ == "__main__":
    main()
