# Dataset Setup

## FakeNewsNet Dataset

This project uses the **FakeNewsNet** dataset, which contains labeled real and fake news articles from multiple sources.

### Dataset Sources

- **Politifact**: Fact-checking articles from politifact.com
- **GossipCop**: Celebrity gossip articles

### Download Instructions

#### Option 1: Automatic Download

```bash
python src/preprocess.py --download
```

#### Option 2: Manual Download

1. Clone the FakeNewsNet repository:
   ```bash
   git clone https://github.com/KaiDMML/FakeNewsNet.git
   ```

2. Extract the dataset files to `data/raw/`

3. Run preprocessing:
   ```bash
   python src/preprocess.py --process
   ```

### Dataset Structure

After preprocessing, the `data/` directory should contain:

```
data/
├── raw/                    # Original FakeNewsNet files
├── processed/
│   ├── train.csv          # Training set (70%)
│   ├── val.csv            # Validation set (15%)
│   └── test.csv           # Test set (15%)
└─┠ README.md
```

### Data Format

Each CSV file contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `id` | str | Unique article identifier |
| `title` | str | Article title |
| `text` | str | Article body text |
| `label` | int | 0 = Real, 1 = Fake |
| `source` | str | Source domain (politifact, gossip, etc) |

### Statistics

**FakeNewsNet Summary:**
- Total articles: ~23,000
- Fake articles: ~12,000 (52%)
- Real articles: ~11,000 (48%)
- Average text length: ~500-800 words

### Class Balance

The dataset is relatively balanced, but with a slight bias toward fake news. The train/val/test split uses stratified sampling to maintain class distribution.

## Alternative Datasets

For multi-domain testing and generalization:

- **FineFake**: Multi-domain fake news with fine-grained annotations
  - Repository: https://github.com/your-repo-here
  - Covers: Politics, Health, Technology, Business, Entertainment

- **Factify-2.0**: Multimodal fake news detection (text + images)
  - Repository: https://github.com/your-repo-here

## Data Privacy & Ethics

- All data is from public sources (Politifact and GossipCop)
- Articles are preprocessed to remove any personally identifiable information
- Use of this data is for research and educational purposes only

## Contact

For dataset-specific issues, visit: https://github.com/KaiDMML/FakeNewsNet
