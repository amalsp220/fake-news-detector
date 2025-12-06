# Fake-News Detection System

An end-to-end machine learning system for automatic fake news detection using transformer models (BERT), baseline ML classifiers, and interpretability tools (SHAP/LIME). Includes dataset preprocessing, model training, evaluation, and a Streamlit web interface for real-time predictions with explainability.

## 📋 Overview

Misinformation and fake news remain a major global problem. This project demonstrates an MLOps-ready pipeline for:

- **Data Preprocessing**: Text cleaning, normalization, and train/val/test splits
- **Baseline Models**: TF-IDF vectorization + Logistic Regression, SVM, Random Forest
- **Advanced Models**: BERT-based fine-tuned transformer for state-of-the-art classification
- **Explainability**: SHAP and LIME integration to understand model decisions
- **Web Interface**: Streamlit app for real-time predictions with highlighted explanations
- **Evaluation**: Comprehensive metrics (accuracy, precision, recall, F1, confusion matrix)

## 🎯 Key Features

✅ **Production-Ready Architecture**: Modular, well-documented codebase  
✅ **Multiple Models**: From classical ML to modern transformers  
✅ **Interpretability**: Understand why the model classified news as fake  
✅ **User-Friendly UI**: Streamlit app for non-technical users  
✅ **Comprehensive Evaluation**: Full metrics and visualization suite  
✅ **Dataset Integration**: Works with FakeNewsNet (Politifact + GossipCop)  

## 📦 Project Structure

```
fake-news-detector/
├── data/
│   └── README.md              # Instructions for dataset download and setup
├── notebooks/
│   └── eda.ipynb             # Exploratory Data Analysis
├── src/
│   ├── preprocess.py         # Text preprocessing and dataset handling
│   ├── models_baseline.py    # Classical ML models (TF-IDF + Logistic/SVM/RF)
│   ├── models_transformer.py # BERT-based transformer model
│   ├── train_baseline.py     # Training script for baseline models
│   ├── train_transformer.py  # Fine-tuning script for BERT
│   └── explain.py            # SHAP/LIME explainability wrappers
├── app/
│   └── streamlit_app.py      # Web interface for predictions
├── requirements.txt           # Python dependencies
├── README.md                 # This file
└── LICENSE                   # MIT License
```

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/amalsp220/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
```

### 2. Dataset Setup

See `data/README.md` for instructions on downloading FakeNewsNet.

### 3. Run Baseline Model

```bash
python src/train_baseline.py
```

### 4. Fine-tune Transformer (BERT)

```bash
python src/train_transformer.py
```

### 5. Launch Web App

```bash
streamlit run app/streamlit_app.py
```

## 📊 Models & Performance

### Baseline Models (Classical ML)

**Models Implemented:**
- **Logistic Regression** with TF-IDF
- **Linear SVM** with TF-IDF
- **Random Forest** with TF-IDF + word embeddings

**Expected Accuracy**: 80–88% on FakeNewsNet

### Advanced Model (Transformer)

**Model**: BERT-base-uncased fine-tuned for binary classification

**Expected Accuracy**: 90–95% on FakeNewsNet

**Advantages**:
- Captures semantic and contextual meaning
- Better generalization to unseen domains
- Built-in attention mechanism for interpretability

## 🔍 Explainability

The project includes two explainability approaches:

### SHAP (SHapley Additive exPlanations)
- Shows contribution of each token to the final prediction
- Global and local explanations

### LIME (Local Interpretable Model-agnostic Explanations)
- Highlights important words influencing classification
- Local approximation of model behavior

**Usage:**

```python
from src.explain import SHAPExplainer, LIMEExplainer

explainer = SHAPExplainer(model)
shap_values = explainer.explain(text_input)
```

## 📈 Evaluation Metrics

Full evaluation pipeline includes:

- **Accuracy**: Overall correctness
- **Precision**: False positive rate (avoid labeling real news as fake)
- **Recall**: False negative rate (catch actual fake news)
- **F1-Score**: Harmonic mean for imbalanced data
- **Confusion Matrix**: Breakdown of predictions
- **ROC-AUC**: Discrimination ability

## 🌐 Web Interface (Streamlit)

**Features:**

- Paste or upload news article text
- Real-time prediction with confidence score
- Highlighted tokens showing "suspicious" vs "trustworthy" words
- Model selection (baseline vs transformer)
- Explainability visualization
- Batch prediction support

## ⚙️ Technologies

**ML & NLP:**
- PyTorch, Transformers (Hugging Face), scikit-learn

**Explainability:**
- SHAP, LIME

**Web:**
- Streamlit

**Data Processing:**
- NumPy, Pandas

**Visualization:**
- Matplotlib, Seaborn

## 🎓 Key Learnings & Extensions

**What You'll Learn:**
1. Full ML pipeline from data to deployment
2. Fine-tuning transformers on custom tasks
3. Model explainability for stakeholder trust
4. Building user-friendly ML applications

**Future Enhancements:**
- Multi-domain evaluation (politics, health, tech)
- Multi-lingual support (non-English languages)
- Multimodal detection (text + images)
- Graph-based approaches (social context)
- Online learning / model drift detection

## ⚠️ Limitations & Ethical Considerations

**Dataset Bias:**
- FakeNewsNet may have language/domain bias
- Model may not generalize perfectly to newer domains or writing styles

**Content-Based Limitations:**
- Purely text-based classification can miss subtle misinformation
- Satire and parody can be misclassified
- Partially true / misleading content is inherently ambiguous

**Ethical Use:**
- Do not use to suppress legitimate criticism or dissent
- Always include human review for high-stakes decisions
- Be transparent about model limitations
- Consider impact on marginalized communities

## 📖 References & Datasets

**Datasets:**
- **FakeNewsNet**: https://github.com/KaiDMML/FakeNewsNet
- **FineFake** (multi-domain): https://arxiv.org/abs/2404.01336
- **Factify-2.0** (multimodal): https://arxiv.org/abs/2304.03897

**Related Work:**
- GNN-FakeNews: https://github.com/safe-graph/GNN-FakeNews
- Knowledge-graph approaches: https://arxiv.org/abs/2107.10648

## 📝 License

MIT License – See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit changes (`git commit -am 'Add my feature'`)
4. Push to branch (`git push origin feature/my-feature`)
5. Open a Pull Request

## 📧 Contact

For questions or discussions, please open an Issue or reach out on GitHub.

---

**Happy detecting! 🎯**
