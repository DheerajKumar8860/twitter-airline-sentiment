# Quick Reference Guide

Fast access to common commands, file locations, and troubleshooting tips.

## ⚡ Quick Commands

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/twitter-airline-sentiment.git
cd twitter-airline-sentiment

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import sklearn, xgboost, transformers; print('Ready!')"
```

### Run Project
```bash
# Step 1: EDA (5-10 minutes)
python src/eda_analysis.py

# Step 2: Train ML models (15-20 minutes)
python src/train_individual.py

# Step 3: Train ensemble (25-35 minutes)
python src/train_ensemble.py
```

### Quick Test
```bash
# Test data loader
python -c "from src.data_loader import load_data; df = load_data(); print(df.shape)"

# Test single model
python -c "from src.train_individual import *; train_svm()"
```

---

## 📁 File Locations

### Input Files
- Dataset: `data/Tweets.csv`
- Source code: `src/*.py`
- Config: `requirements.txt`

### Output Files
- EDA plots: `outputs/eda/*.png`
- Metrics: `outputs/metrics_comparison.csv`
- Comparison plot: `outputs/metrics_comparison.png`
- Models: `models/*.pkl`

### Documentation
- Main README: `README.md`
- Setup guide: `SETUP_GUIDE.md`
- Technical details: `PROJECT_SUMMARY.md`
- This file: `QUICK_REFERENCE.md`

---

## 🔧 Common Issues

### Import Error
```bash
# Fix: Reinstall packages
pip install --upgrade pip
pip install -r requirements.txt
```

### Data Not Found
```bash
# Fix: Check file location
ls data/Tweets.csv
# Should exist, if not download from Kaggle
```

### Slow Training
```bash
# Fix: Use smaller dataset for testing
# Edit data_loader.py, add: df = df.sample(1000)
```

### Out of Memory
```bash
# Fix: Reduce batch size in train_ensemble.py
# Change: batch_size = 8 (instead of 16)
```

---

## 📊 Expected Results

| Metric | SVM | XGBoost | DistilBERT | Ensemble |
|--------|-----|---------|------------|----------|
| Accuracy | 0.78 | 0.80 | 0.85 | 0.87 |
| Training Time | 3-5 min | 5-10 min | 20-30 min | 30-40 min |
| Inference | 5ms | 8ms | 45ms | 58ms |

---

## 🎯 Key Functions

### data_loader.py
- `load_data()` - Load CSV dataset
- `preprocess_text(text)` - Clean tweet text
- `split_data(df)` - Train-test split

### eda_analysis.py
- `plot_sentiment_distribution()` - Class balance
- `plot_text_length()` - Length analysis
- `generate_wordclouds()` - Visual keywords

### train_individual.py
- `generate_embeddings()` - Sentence Transformers
- `train_svm()` - Train SVM classifier
- `train_xgboost()` - Train XGBoost classifier

### train_ensemble.py
- `fine_tune_bert()` - Train DistilBERT
- `create_ensemble()` - Combine predictions
- `evaluate_models()` - Compare performance

---

## 🌐 Useful Links

- **Kaggle Dataset**: [Twitter Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- **Sentence Transformers**: [Documentation](https://huggingface.co/sentence-transformers)
- **DistilBERT**: [Model Card](https://huggingface.co/distilbert-base-uncased)
- **XGBoost Docs**: [xgboost.readthedocs.io](https://xgboost.readthedocs.io)
- **Scikit-learn**: [scikit-learn.org](https://scikit-learn.org)

---

## 💡 Tips & Tricks

### Speed Up Training
1. Use GPU for DistilBERT: `device='cuda'`
2. Reduce epochs: `epochs=2` instead of 3
3. Smaller dataset: Use 50% sample for testing

### Improve Accuracy
1. Tune hyperparameters: Grid search
2. Balance classes: Use class weights
3. Add features: Tweet metadata (time, airline)
4. Ensemble weights: Experiment with different ratios

### Debug Issues
1. Add print statements: Track progress
2. Test small samples: Validate pipeline
3. Check shapes: `print(X.shape, y.shape)`
4. Verify data: `df.head()`, `df.info()`

### Save Time
1. Cache embeddings: Save to disk after generation
2. Use pre-trained: Don't retrain if model exists
3. Parallel processing: Use all CPU cores
4. Early stopping: Stop if no improvement

---

## 🚀 Next Steps

### Beginners
1. Run all scripts successfully
2. Understand each visualization
3. Read through code comments
4. Experiment with small changes

### Intermediate
1. Tune hyperparameters
2. Add new visualizations
3. Try different embeddings
4. Implement cross-validation

### Advanced
1. Deploy as API (FastAPI)
2. Add real-time streaming
3. Implement MLOps pipeline
4. Build web dashboard

---

## 📞 Get Help

1. Check error message carefully
2. Search issue in GitHub
3. Ask on Stack Overflow
4. Read documentation
5. Open issue in repository

---

## ✅ Pre-flight Checklist

Before running:
- [ ] Python 3.8+ installed
- [ ] Dependencies installed
- [ ] Dataset downloaded
- [ ] All folders created
- [ ] 2GB+ free space
- [ ] Internet connected

---

*Keep this guide handy for quick reference!*