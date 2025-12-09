# Setup Guide - Twitter Airline Sentiment Analysis

This guide provides detailed, step-by-step instructions for setting up and running the project, including manual GitHub repository creation.

## 📦 Table of Contents

1. [System Requirements](#system-requirements)
2. [Manual GitHub Repository Setup](#manual-github-repository-setup)
3. [Local Environment Setup](#local-environment-setup)
4. [Dataset Download](#dataset-download)
5. [Running the Project](#running-the-project)
6. [Troubleshooting](#troubleshooting)

---

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB free space
- **Internet**: Required for downloading models and datasets

### Software Dependencies
- Python 3.8+
- pip (Python package installer)
- Git (optional, for cloning)

---

## 🌐 Manual GitHub Repository Setup

Follow these steps to create your repository manually without using terminal commands:

### Step 1: Create the Repository on GitHub

1. **Log in to GitHub**
   - Go to [github.com](https://github.com)
   - Sign in to your account

2. **Create New Repository**
   - Click the `+` icon in the top-right corner
   - Select `New repository`

3. **Configure Repository**
   - **Repository name**: `twitter-airline-sentiment`
   - **Description**: "Hybrid sentiment analysis using ML and transformers"
   - **Visibility**: Public (recommended for portfolio)
   - **Initialize**: Check "Add a README file"
   - Click `Create repository`

### Step 2: Create Folder Structure

GitHub doesn't allow creating empty folders, so we'll create folders by adding files to them.

#### Create `src/` folder:

1. In your repository, click `Add file` → `Create new file`
2. Type: `src/__init__.py`
3. Leave the file empty or add a comment: `# Source code package`
4. Scroll down, add commit message: "Create src folder"
5. Click `Commit new file`

#### Create `data/` folder:

1. Click `Add file` → `Create new file`
2. Type: `data/.gitkeep`
3. Add comment: `# Placeholder for data folder`
4. Commit with message: "Create data folder"

#### Create `outputs/eda/` folder:

1. Click `Add file` → `Create new file`
2. Type: `outputs/eda/.gitkeep`
3. Commit with message: "Create outputs folder structure"

#### Create `models/` folder:

1. Click `Add file` → `Create new file`
2. Type: `models/.gitkeep`
3. Commit with message: "Create models folder"

### Step 3: Upload Documentation Files

For each documentation file (README.md, SETUP_GUIDE.md, etc.):

1. Click `Add file` → `Create new file`
2. Name the file (e.g., `SETUP_GUIDE.md`)
3. Copy and paste the content provided in this guide
4. Commit with a descriptive message
5. Repeat for all documentation files

### Step 4: Upload Source Code Files

For each Python file in `src/`:

1. Navigate to the `src/` folder in your repository
2. Click `Add file` → `Create new file`
3. Name the file (e.g., `data_loader.py`)
4. Copy and paste the code provided in this guide
5. Commit with message like "Add data loader module"
6. Repeat for all source files

### Step 5: Upload Configuration Files

1. Create `requirements.txt`:
   - Click `Add file` → `Create new file`
   - Name: `requirements.txt`
   - Paste dependencies
   - Commit

2. Create `.gitignore`:
   - Click `Add file` → `Create new file`
   - Name: `.gitignore`
   - Paste ignore rules
   - Commit

---

## 💻 Local Environment Setup

### Option 1: Using pip (Recommended)

1. **Open Terminal/Command Prompt**
   - Windows: Press `Win + R`, type `cmd`, press Enter
   - macOS: Press `Cmd + Space`, type `terminal`, press Enter
   - Linux: Press `Ctrl + Alt + T`

2. **Navigate to Project Directory**
   ```bash
   cd path/to/twitter-airline-sentiment
   ```

3. **Create Virtual Environment** (Optional but recommended)
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify Installation**
   ```bash
   python -c "import sklearn, xgboost, transformers; print('All packages installed!')"
   ```

### Option 2: Using Conda

1. **Create Conda Environment**
   ```bash
   conda create -n sentiment python=3.9
   conda activate sentiment
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📥 Dataset Download

### Method 1: Manual Download from Kaggle

1. **Visit Kaggle Dataset**
   - Go to: [https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)

2. **Sign in to Kaggle**
   - Create a free account if you don't have one

3. **Download Dataset**
   - Click the `Download` button (may require phone verification)
   - This downloads `twitter-airline-sentiment.zip`

4. **Extract and Move File**
   - Unzip the downloaded file
   - Find `Tweets.csv` inside
   - Move `Tweets.csv` to your `data/` folder

5. **Verify File Location**
   - Ensure the path is: `twitter-airline-sentiment/data/Tweets.csv`

### Method 2: Using Kaggle API (Advanced)

1. **Install Kaggle API**
   ```bash
   pip install kaggle
   ```

2. **Configure API Credentials**
   - Go to Kaggle Account Settings
   - Scroll to API section
   - Click "Create New API Token"
   - Save `kaggle.json` to:
     - Windows: `C:\Users\<YourUsername>\.kaggle\`
     - macOS/Linux: `~/.kaggle/`

3. **Download Dataset**
   ```bash
   cd data/
   kaggle datasets download -d crowdflower/twitter-airline-sentiment
   unzip twitter-airline-sentiment.zip
   rm twitter-airline-sentiment.zip
   ```

---

## ▶️ Running the Project

### Step 1: Exploratory Data Analysis (EDA)

```bash
python src/eda_analysis.py
```

**What happens:**
- Loads `Tweets.csv` from `data/` folder
- Generates 5+ visualizations
- Saves plots to `outputs/eda/`
- Prints summary statistics

**Expected output:**
```
Loading dataset...
Dataset loaded: 14640 tweets
Performing EDA...
✓ Sentiment distribution plot saved
✓ Text length analysis saved
✓ Word clouds generated
✓ Airline comparison saved
✓ Temporal analysis saved
EDA complete!
```

**Check results:**
- Navigate to `outputs/eda/` folder
- View generated PNG files

### Step 2: Train Individual ML Models

```bash
python src/train_individual.py
```

**What happens:**
- Loads and preprocesses data
- Generates Sentence Transformer embeddings
- Trains SVM model
- Trains XGBoost model
- Evaluates both models
- Saves models to `models/` folder

**Expected output:**
```
Loading dataset...
Preprocessing text...
Generating embeddings...
Training SVM...
  SVM Accuracy: 0.78
  SVM F1-Score: 0.77
Training XGBoost...
  XGBoost Accuracy: 0.80
  XGBoost F1-Score: 0.79
Models saved successfully!
```

**Training time:**
- SVM: 2-5 minutes
- XGBoost: 5-10 minutes
- Total: ~15 minutes

### Step 3: Train Ensemble Model

```bash
python src/train_ensemble.py
```

**What happens:**
- Loads pre-trained SVM and XGBoost models
- Fine-tunes DistilBERT transformer
- Creates ensemble predictions
- Saves performance metrics
- Generates comparison visualizations

**Expected output:**
```
Loading individual models...
Fine-tuning DistilBERT...
Epoch 1/3: loss=0.45
Epoch 2/3: loss=0.28
Epoch 3/3: loss=0.18
Creating ensemble predictions...
  Ensemble Accuracy: 0.87
  Ensemble F1-Score: 0.86
Results saved to outputs/
```

**Training time:**
- DistilBERT fine-tuning: 20-30 minutes (CPU), 5-10 minutes (GPU)
- Ensemble creation: 2-3 minutes
- Total: ~25-35 minutes

### Step 4: View Results

**Metrics Comparison:**
- Open `outputs/metrics_comparison.csv` in Excel/Google Sheets
- View `outputs/metrics_comparison.png` for visual comparison

**Model Files:**
- `models/svm_model.pkl` - Trained SVM classifier
- `models/xgb_model.pkl` - Trained XGBoost classifier
- DistilBERT saved in HuggingFace cache

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue 1: ModuleNotFoundError

**Error:**
```
ModuleNotFoundError: No module named 'transformers'
```

**Solution:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Issue 2: FileNotFoundError for Dataset

**Error:**
```
FileNotFoundError: data/Tweets.csv not found
```

**Solution:**
- Verify `Tweets.csv` is in the `data/` folder
- Check file name is exactly `Tweets.csv` (case-sensitive)
- Re-download from Kaggle if necessary

#### Issue 3: CUDA/GPU Issues

**Error:**
```
CUDA out of memory
```

**Solution:**
- Reduce batch size in `train_ensemble.py`
- Or force CPU usage by setting:
  ```python
  device = 'cpu'
  ```

#### Issue 4: Slow Training

**Symptoms:**
- Training takes hours instead of minutes

**Solutions:**
1. **Use GPU** (if available):
   - Install PyTorch with CUDA support
   - Verify GPU is detected: `torch.cuda.is_available()`

2. **Reduce dataset size** (for testing):
   - Modify `data_loader.py` to use smaller sample
   - Use only 10-20% of data for quick testing

3. **Use smaller model**:
   - Replace DistilBERT with a smaller variant
   - Consider using only ML models first

#### Issue 5: Memory Errors

**Error:**
```
MemoryError: Unable to allocate array
```

**Solutions:**
1. Close other applications
2. Reduce batch size
3. Process data in chunks
4. Use a machine with more RAM

### Getting Help

If you encounter issues not covered here:

1. **Check error messages carefully** - they often indicate the exact problem
2. **Search GitHub Issues** - someone may have faced the same issue
3. **Ask on Stack Overflow** - tag with `python`, `scikit-learn`, `transformers`
4. **Open an issue** - in this repository with details about your problem

---

## ✅ Verification Checklist

Before running the project, verify:

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip list` shows all packages)
- [ ] `Tweets.csv` in `data/` folder
- [ ] All folders created: `src/`, `data/`, `outputs/`, `models/`
- [ ] All Python files uploaded to `src/`
- [ ] At least 2GB free disk space
- [ ] Internet connection available (for first run)

---

## 🎓 Next Steps

After successful setup:

1. **Explore the code** - Understand each module's purpose
2. **Experiment** - Modify hyperparameters, try different models
3. **Extend** - Add new features, try other datasets
4. **Document** - Record your findings and improvements
5. **Share** - Show off your work on LinkedIn/portfolio

Happy coding! 🚀