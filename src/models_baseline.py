"""Baseline ML models for fake news detection using classical methods."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from typing import Tuple


class BaselineModel:
    """Wrapper for classical ML models."""
    
    def __init__(self, model_type: str = 'logistic', vectorizer: TfidfVectorizer = None):
        self.model_type = model_type
        self.vectorizer = vectorizer or TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        
        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'svm':
            self.model = LinearSVC(max_iter=2000, random_state=42)
        elif model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'BaselineModel':
        """Fit the model."""
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For SVM, use decision function
            scores = self.model.decision_function(X)
            return np.column_stack([(1 - scores) / 2, (1 + scores) / 2])
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        joblib.dump(self.model, f"{path}_model.pkl")
        joblib.dump(self.vectorizer, f"{path}_vectorizer.pkl")
    
    @staticmethod
    def load(path: str) -> 'BaselineModel':
        """Load model from disk."""
        model_obj = BaselineModel()
        model_obj.model = joblib.load(f"{path}_model.pkl")
        model_obj.vectorizer = joblib.load(f"{path}_vectorizer.pkl")
        return model_obj


def train_baseline(X_train: pd.Series, X_val: pd.Series, y_train: np.ndarray, 
                   y_val: np.ndarray, model_type: str = 'logistic') -> Tuple[BaselineModel, dict]:
    """Train and evaluate baseline model."""
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_val_vec = vectorizer.transform(X_val).toarray()
    
    # Train
    model = BaselineModel(model_type, vectorizer)
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val_vec)
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'confusion_matrix': confusion_matrix(y_val, y_pred).tolist()
    }
    
    return model, metrics


if __name__ == "__main__":
    import sys
    from preprocess import FakeNewsDataset
    
    print("Loading data...")
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    
    model, metrics = train_baseline(
        train_df['full_text'], val_df['full_text'],
        train_df['label'].values, val_df['label'].values,
        model_type='logistic'
    )
    
    print(f"Baseline Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
