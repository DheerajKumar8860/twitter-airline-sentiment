"""
Data loading and preprocessing utilities for Twitter sentiment analysis.
"""

import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
from typing import Tuple


def load_data(filepath: str = 'data/Tweets.csv') -> pd.DataFrame:
    """
    Load the Twitter airline sentiment dataset.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with tweet text and sentiment labels
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Select relevant columns
    df = df[['text', 'airline_sentiment']].copy()
    
    # Rename for clarity
    df.columns = ['text', 'sentiment']
    
    # Remove any missing values
    df = df.dropna()
    
    print(f"Dataset loaded: {len(df)} tweets")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    
    return df


def preprocess_text(text: str) -> str:
    """
    Clean and preprocess tweet text while preserving sentiment indicators.
    
    Args:
        text: Raw tweet text
        
    Returns:
        Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Replace @mentions with generic token
    text = re.sub(r'@\w+', '@user', text)
    
    # Remove hashtag symbol but keep the text
    text = re.sub(r'#', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing to all texts in the dataframe.
    
    Args:
        df: DataFrame with 'text' column
        
    Returns:
        DataFrame with preprocessed text
    """
    print("Preprocessing text data...")
    df['text'] = df['text'].apply(preprocess_text)
    
    # Remove any empty texts after preprocessing
    df = df[df['text'].str.len() > 0]
    
    print(f"Preprocessing complete. {len(df)} tweets remain.")
    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert sentiment labels to numeric format.
    
    Args:
        df: DataFrame with 'sentiment' column
        
    Returns:
        DataFrame with numeric labels
    """
    label_map = {
        'negative': 0,
        'neutral': 1,
        'positive': 2
    }
    
    df['label'] = df['sentiment'].map(label_map)
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, 
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets with stratification.
    
    Args:
        df: DataFrame with text and labels
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['label']
    )
    
    print(f"\nData split complete:")
    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")
    
    return train_df, test_df


def prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete data preparation pipeline.
    
    Returns:
        Tuple of (train_df, test_df) ready for modeling
    """
    # Load data
    df = load_data()
    
    # Preprocess
    df = preprocess_dataframe(df)
    
    # Encode labels
    df = encode_labels(df)
    
    # Split data
    train_df, test_df = split_data(df)
    
    return train_df, test_df


if __name__ == "__main__":
    # Test the data loader
    train_df, test_df = prepare_data()
    
    print("\n" + "="*50)
    print("Sample preprocessed tweets:")
    print("="*50)
    for i, row in train_df.head(3).iterrows():
        print(f"\nSentiment: {row['sentiment']}")
        print(f"Text: {row['text']}")
    
    print("\n" + "="*50)
    print("Data loading test successful!")
    print("="*50)