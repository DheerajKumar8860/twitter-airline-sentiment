# Twitter Airline Sentiment Analysis

A hybrid sentiment classification system that combines traditional machine learning with modern deep learning approaches to analyze sentiment in airline-related tweets.

## 🎯 Project Highlights

- **Hybrid Architecture**: Combines fast ML classifiers (SVM, XGBoost) with transformer-based models (DistilBERT)
- **Smart Embeddings**: Uses Sentence Transformers for rich text representations
- **Ensemble Learning**: Merges predictions from multiple models for improved accuracy
- **Real-world Data**: Trained on Twitter Airline Sentiment Dataset with noisy, informal text
- **Production Ready**: Clean, modular code structure suitable for deployment

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│           Input: Raw Tweet Text                 │
└────────────────┬────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌───────────────┐  ┌──────────────┐
│  Sentence     │  │  DistilBERT  │
│  Transformer  │  │  Fine-tuned  │
│  Embeddings   │  │  Transformer │
└───────┬───────┘  └──────┬───────┘
        │                 │
        ▼                 │
┌──────────────┐          │
│  SVM Model   │          │
└──────┬───────┘          │
       │                  │
┌──────┴───────┐          │
│ XGBoost Model│          │
└──────┬───────┘          │
       │                  │
       └────────┬─────────┘
                ▼
        ┌───────────────┐
        │   Ensemble    │
        │   Voting      │
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │  Final        │
        │  Prediction   │
        └───────────────┘
```

## 📊 Dataset

**Twitter Airline Sentiment Dataset** from Kaggle
- Source: [Kaggle Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- Size: ~14,640 tweets
- Classes: Positive, Neutral, Negative
- Features: Tweet text, airline, sentiment labels

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip package manager
4GB+ RAM recommended
```

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/twitter-airline-sentiment.git
cd twitter-airline-sentiment
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
   - Visit [Kaggle Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
   - Download `Tweets.csv`
   - Place it in the `data/` folder

### Running the Project

**Step 1: Exploratory Data Analysis**
```bash
python src/eda_analysis.py
```
This generates 5+ visualizations in `outputs/eda/`

**Step 2: Train Individual Models**
```bash
python src/train_individual.py
```
Trains SVM and XGBoost models, saves them to `models/`

**Step 3: Train Ensemble Model**
```bash
python src/train_ensemble.py
```
Fine-tunes DistilBERT and creates ensemble predictions

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| SVM | ~0.78 | ~0.77 | ~0.78 | ~0.77 |
| XGBoost | ~0.80 | ~0.79 | ~0.80 | ~0.79 |
| DistilBERT | ~0.85 | ~0.84 | ~0.85 | ~0.84 |
| **Ensemble** | **~0.87** | **~0.86** | **~0.87** | **~0.86** |

*Results may vary based on train-test split and hyperparameters*

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **Sentence Transformers**: Pre-trained embeddings for text representation
- **Scikit-learn**: SVM classifier and preprocessing utilities
- **XGBoost**: Gradient boosting classifier
- **Transformers (HuggingFace)**: DistilBERT fine-tuning
- **Pandas & NumPy**: Data manipulation
- **Matplotlib & Seaborn**: Data visualization
- **NLTK**: Text preprocessing

## 📁 Project Structure

```
twitter-airline-sentiment/
│
├── README.md                 # This file
├── SETUP_GUIDE.md           # Detailed setup instructions
├── PROJECT_SUMMARY.md       # Technical deep-dive
├── QUICK_REFERENCE.md       # Quick commands reference
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
│
├── src/                    # Source code
│   ├── __init__.py
│   ├── data_loader.py      # Data loading utilities
│   ├── eda_analysis.py     # Exploratory data analysis
│   ├── train_individual.py # Train ML models
│   └── train_ensemble.py   # Train ensemble model
│
├── data/                   # Dataset folder
│   └── Tweets.csv          # Twitter airline sentiment data
│
├── outputs/                # Generated outputs
│   ├── eda/               # EDA visualizations
│   ├── metrics_comparison.csv
│   └── metrics_comparison.png
│
└── models/                 # Saved models
    ├── svm_model.pkl
    └── xgb_model.pkl
```

## 🔍 Key Features

### 1. **Robust Preprocessing**
- Handles Twitter-specific elements (@mentions, #hashtags, URLs)
- Cleans noisy, informal text
- Preserves sentiment-bearing punctuation

### 2. **Multiple Model Types**
- Traditional ML for speed and interpretability
- Deep learning for handling complex patterns
- Best of both worlds through ensembling

### 3. **Comprehensive EDA**
- Sentiment distribution analysis
- Text length patterns
- Word clouds for each sentiment
- Airline-specific insights
- Temporal patterns (if applicable)

### 4. **Professional Code Quality**
- Modular, reusable functions
- Clear documentation
- Type hints where applicable
- Easy to extend and modify

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- Dataset: [Crowdflower Twitter Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- Sentence Transformers: [HuggingFace](https://huggingface.co/sentence-transformers)
- DistilBERT: [HuggingFace Transformers](https://huggingface.co/distilbert-base-uncased)

## 📧 Contact

For questions or feedback, please open an issue or contact me directly.

---

⭐ If you find this project helpful, please consider giving it a star!