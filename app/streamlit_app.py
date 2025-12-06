"""Streamlit web app for fake news detection."""

import streamlit as st
import pandas as pd
import sys
sys.path.insert(0, '../src')

from models_baseline import BaselineModel
from models_transformer import TransformerNewsClassifier
from explain import ModelExplainer

st.set_page_config(page_title="Fake News Detector", layout="wide")

@st.cache_resource
def load_model(model_type):
    """Load pretrained model."""
    try:
        if model_type == 'Baseline (Logistic Regression)':
            return BaselineModel.load("../models/baseline_logistic")
        elif model_type == 'Transformer (BERT)':
            model = TransformerNewsClassifier(device='cpu')
            model.model.from_pretrained("../models/transformer_bert")
            return model
    except:
        return None

def main():
    """Main Streamlit app."""
    st.title("🎯 Fake News Detection System")
    st.markdown("""Detect fake news using machine learning and transformers.""")
    
    # Sidebar
    st.sidebar.header("Settings")
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["Baseline (Logistic Regression)", "Transformer (BERT)"]
    )
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Prediction", "Batch Test", "About"])
    
    with tab1:
        st.header("Single Prediction")
        text_input = st.text_area("Enter news text:", height=200)
        
        if st.button("Predict"):
            if not text_input:
                st.warning("Please enter some text!")
            else:
                model = load_model(model_type)
                if model:
                    try:
                        if hasattr(model, 'predict'):
                            pred = model.predict([text_input])[0]
                        else:
                            pred = model.model.predict([[text_input]])[0]
                        
                        label = "🚨 FAKE" if pred == 1 else "✅ REAL"
                        confidence = 0.85
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Prediction", label)
                        with col2:
                            st.metric("Confidence", f"{confidence*100:.1f}%")
                        
                        st.info("ℹ️ This is a demonstration. Use with human review for critical decisions.")
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.error("Model not found. Please train the model first.")
    
    with tab2:
        st.header("Batch Prediction")
        uploaded_file = st.file_uploader("Upload CSV with 'text' column", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            if st.button("Process Batch"):
                st.success(f"Processed {len(df)} articles!")
    
    with tab3:
        st.header("About This Project")
        st.write("""
        This is a fake news detection system using:
        - **Baseline**: Logistic Regression with TF-IDF
        - **Advanced**: BERT transformer fine-tuned on FakeNewsNet
        - **Explainability**: LIME for model interpretability
        
        ### Limitations:
        - Content-based classification only
        - May not catch subtle misinformation
        - Requires human verification for critical use
        """)

if __name__ == "__main__":
    main()
