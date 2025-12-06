"""Model explainability using SHAP and LIME."""

import numpy as np
import shap
from lime.lime_text import LimeTextExplainer
from typing import Dict, List, Tuple


class ModelExplainer:
    """Explainability wrapper for fake news models."""
    
    def __init__(self, model, tokenizer=None, model_type='transformer'):
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        
        if model_type == 'transformer':
            self.lime_explainer = LimeTextExplainer(class_names=['Real', 'Fake'])
        else:
            self.lime_explainer = LimeTextExplainer(class_names=['Real', 'Fake'])
    
    def predict_fn_baseline(self, texts):
        """Prediction function for baseline models."""
        from models_baseline import BaselineModel
        if isinstance(texts, list):
            texts = np.array(texts)
        vec = self.model.vectorizer.transform(texts)
        probs = self.model.predict_proba(vec.toarray())
        return probs
    
    def predict_fn_transformer(self, texts):
        """Prediction function for transformer models."""
        preds = self.model.predict(texts)
        probs = np.zeros((len(texts), 2))
        probs[np.arange(len(texts)), preds] = 1.0
        return probs
    
    def explain_lime(self, text: str, num_features: int = 10) -> Dict:
        """Get LIME explanation."""
        if self.model_type == 'transformer':
            predict_fn = self.predict_fn_transformer
        else:
            predict_fn = self.predict_fn_baseline
        
        exp = self.lime_explainer.explain_instance(
            text, predict_fn, num_features=num_features, top_labels=1
        )
        
        label = exp.available_labels()[0]
        word_weights = dict(exp.as_list(label=label))
        
        return {
            'label': label,
            'label_name': 'Fake' if label == 1 else 'Real',
            'word_weights': word_weights,
            'confidence': exp.predict_proba[label]
        }
    
    def explain_text_tokens(self, text: str) -> Tuple[List[str], List[float]]:
        """Highlight important tokens in text."""
        explanation = self.explain_lime(text, num_features=20)
        
        words_weights = explanation['word_weights']
        tokens = text.lower().split()
        
        token_importances = []
        for token in tokens:
            weight = 0
            for word, w in words_weights.items():
                if word.lower() in token.lower():
                    weight += w
            token_importances.append(weight)
        
        return tokens, token_importances


def visualize_explanation(explanation: Dict) -> str:
    """Create HTML visualization of explanation."""
    html = f"<h3>Prediction: {explanation['label_name']}</h3>"
    html += f"<p>Confidence: {explanation['confidence']:.2%}</p>"
    html += "<h4>Important Words:</h4><ul>"
    
    for word, weight in sorted(explanation['word_weights'].items(), 
                               key=lambda x: abs(x[1]), reverse=True):
        color = 'green' if weight > 0 else 'red'
        html += f"<li><span style='color:{color}'>{word}</span>: {weight:.3f}</li>"
    
    html += "</ul>"
    return html
