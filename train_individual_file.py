"""
Train individual ML models (SVM and XGBoost) using Sentence Transformer embeddings.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
import xgboost as xgb
import joblib
import os
from data_loader import prepare_data


def create_output_dirs():
    """Create necessary output directories."""
    os.makedirs('models', exist_ok=True)
    print("Output directories created.\n")


def generate_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    """
    Generate sentence embeddings using Sentence Transformers.
    
    Args:
        texts: List of text strings
        model_name: Pre-trained model to use
        
    Returns:
        Numpy array of embeddings
    """
    print(f"Loading Sentence Transformer model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def train_svm(X_train, y_train, X_test, y_test):
    """
    Train and evaluate SVM classifier.
    
    Args:
        X_train, y_train: Training data and labels
        X_test, y_test: Test data and labels
        
    Returns:
        Trained SVM model
    """
    print("\n" + "="*60)
    print("TRAINING SVM CLASSIFIER")
    print("="*60)
    
    # Initialize model
    svm_model = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    
    # Train
    print("Training SVM...")
    svm_model.fit(X_train, y_train)
    
    # Predict
    print("Generating predictions...")
    y_pred = svm_model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n--- SVM Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Negative', 'Neutral', 'Positive']))
    
    # Save model
    model_path = 'models/svm_model.pkl'
    joblib.dump(svm_model, model_path)
    print(f"\n✓ SVM model saved to: {model_path}")
    
    return svm_model


def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train and evaluate XGBoost classifier.
    
    Args:
        X_train, y_train: Training data and labels
        X_test, y_test: Test data and labels
        
    Returns:
        Trained XGBoost model
    """
    print("\n" + "="*60)
    print("TRAINING XGBOOST CLASSIFIER")
    print("="*60)
    
    # Initialize model
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        tree_method='hist'
    )
    
    # Train
    print("Training XGBoost...")
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Predict
    print("Generating predictions...")
    y_pred = xgb_model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n--- XGBoost Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Negative', 'Neutral', 'Positive']))
    
    # Save model
    model_path = 'models/xgb_model.pkl'
    joblib.dump(xgb_model, model_path)
    print(f"\n✓ XGBoost model saved to: {model_path}")
    
    return xgb_model


def save_embeddings(train_embeddings, test_embeddings):
    """
    Save embeddings for later use in ensemble model.
    
    Args:
        train_embeddings: Training embeddings
        test_embeddings: Test embeddings
    """
    np.save('models/train_embeddings.npy', train_embeddings)
    np.save('models/test_embeddings.npy', test_embeddings)
    print("\n✓ Embeddings saved for ensemble model")


def main():
    """
    Main training pipeline for individual ML models.
    """
    print("\n" + "="*70)
    print("TRAINING INDIVIDUAL ML MODELS - SVM & XGBOOST")
    print("="*70 + "\n")
    
    # Create output directories
    create_output_dirs()
    
    # Prepare data
    print("Step 1: Loading and preprocessing data...")
    train_df, test_df = prepare_data()
    
    # Generate embeddings
    print("\nStep 2: Generating Sentence Transformer embeddings...")
    train_embeddings = generate_embeddings(train_df['text'].tolist())
    test_embeddings = generate_embeddings(test_df['text'].tolist())
    
    # Save embeddings
    save_embeddings(train_embeddings, test_embeddings)
    
    # Prepare labels
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    # Train SVM
    print("\nStep 3: Training SVM classifier...")
    svm_model = train_svm(train_embeddings, y_train, test_embeddings, y_test)
    
    # Train XGBoost
    print("\nStep 4: Training XGBoost classifier...")
    xgb_model = train_xgboost(train_embeddings, y_train, test_embeddings, y_test)
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nModels saved:")
    print("  ✓ models/svm_model.pkl")
    print("  ✓ models/xgb_model.pkl")
    print("  ✓ models/train_embeddings.npy")
    print("  ✓ models/test_embeddings.npy")
    print("\nNext step: Run train_ensemble.py to create the ensemble model")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()