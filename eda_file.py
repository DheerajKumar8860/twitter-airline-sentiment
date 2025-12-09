"""
Exploratory Data Analysis for Twitter Airline Sentiment Dataset.
Generates comprehensive visualizations to understand the data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
from collections import Counter
import re


# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def create_output_dir():
    """Create output directory for EDA visualizations."""
    os.makedirs('outputs/eda', exist_ok=True)
    print("Output directory created: outputs/eda/")


def load_dataset():
    """Load the raw dataset."""
    print("Loading dataset...")
    df = pd.read_csv('data/Tweets.csv')
    print(f"Dataset loaded: {len(df)} tweets\n")
    return df


def plot_sentiment_distribution(df: pd.DataFrame):
    """
    Plot the distribution of sentiments in the dataset.
    Shows class imbalance clearly.
    """
    print("Creating sentiment distribution plot...")
    
    plt.figure(figsize=(10, 6))
    sentiment_counts = df['airline_sentiment'].value_counts()
    
    colors = ['#d62728', '#ff7f0e', '#2ca02c']  # Red, Orange, Green
    ax = sentiment_counts.plot(kind='bar', color=colors, edgecolor='black', linewidth=1.5)
    
    plt.title('Sentiment Distribution in Twitter Airline Dataset', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Sentiment', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Tweets', fontsize=12, fontweight='bold')
    plt.xticks(rotation=0)
    
    # Add count labels on bars
    for i, v in enumerate(sentiment_counts):
        ax.text(i, v + 200, f'{v:,}\n({v/len(df)*100:.1f}%)', 
                ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/eda/sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Sentiment distribution plot saved\n")


def plot_text_length_analysis(df: pd.DataFrame):
    """
    Analyze and visualize tweet length patterns by sentiment.
    """
    print("Creating text length analysis...")
    
    df['text_length'] = df['text'].str.len()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    sentiment_order = ['negative', 'neutral', 'positive']
    colors = ['#d62728', '#ff7f0e', '#2ca02c']
    
    sns.boxplot(data=df, x='airline_sentiment', y='text_length', 
                order=sentiment_order, palette=colors, ax=axes[0])
    axes[0].set_title('Tweet Length Distribution by Sentiment', 
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Sentiment', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Character Count', fontsize=12, fontweight='bold')
    
    # Histogram
    for sentiment, color in zip(sentiment_order, colors):
        subset = df[df['airline_sentiment'] == sentiment]['text_length']
        axes[1].hist(subset, bins=30, alpha=0.6, label=sentiment.capitalize(), 
                     color=color, edgecolor='black')
    
    axes[1].set_title('Tweet Length Histogram by Sentiment', 
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Character Count', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('outputs/eda/text_length_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Text length analysis saved\n")


def generate_wordclouds(df: pd.DataFrame):
    """
    Create word clouds for each sentiment category.
    """
    print("Generating word clouds for each sentiment...")
    
    sentiments = ['negative', 'neutral', 'positive']
    colors = ['Reds', 'Oranges', 'Greens']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (sentiment, cmap) in enumerate(zip(sentiments, colors)):
        text = ' '.join(df[df['airline_sentiment'] == sentiment]['text'].values)
        
        wordcloud = WordCloud(
            width=600, 
            height=400,
            background_color='white',
            colormap=cmap,
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(text)
        
        axes[idx].imshow(wordcloud, interpolation='bilinear')
        axes[idx].set_title(f'{sentiment.capitalize()} Sentiment', 
                           fontsize=14, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/eda/wordclouds_by_sentiment.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Word clouds saved\n")


def plot_airline_comparison(df: pd.DataFrame):
    """
    Compare sentiment distribution across different airlines.
    """
    print("Creating airline comparison plot...")
    
    # Create crosstab
    airline_sentiment = pd.crosstab(df['airline'], df['airline_sentiment'], 
                                     normalize='index') * 100
    
    airline_sentiment = airline_sentiment[['negative', 'neutral', 'positive']]
    
    ax = airline_sentiment.plot(kind='bar', stacked=True, 
                                 color=['#d62728', '#ff7f0e', '#2ca02c'],
                                 figsize=(12, 6), edgecolor='black', linewidth=1)
    
    plt.title('Sentiment Distribution by Airline', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Airline', fontsize=12, fontweight='bold')
    plt.ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Sentiment', title_fontsize=12, fontsize=10)
    plt.tight_layout()
    
    plt.savefig('outputs/eda/airline_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Airline comparison saved\n")


def plot_top_words(df: pd.DataFrame):
    """
    Show the most common words for each sentiment.
    """
    print("Analyzing top words by sentiment...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sentiments = ['negative', 'neutral', 'positive']
    colors = ['#d62728', '#ff7f0e', '#2ca02c']
    
    for idx, (sentiment, color) in enumerate(zip(sentiments, colors)):
        # Get all text for this sentiment
        text = ' '.join(df[df['airline_sentiment'] == sentiment]['text'].values)
        
        # Simple tokenization
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'to', 'of', 'and', 'in', 'is', 'it', 
                     'for', 'on', 'at', 'with', 'my', 'i', 'was', 'am', 'are'}
        words = [w for w in words if w not in stopwords and len(w) > 2]
        
        # Count most common
        word_counts = Counter(words).most_common(15)
        words_list, counts = zip(*word_counts)
        
        axes[idx].barh(range(len(words_list)), counts, color=color, 
                       edgecolor='black', linewidth=1)
        axes[idx].set_yticks(range(len(words_list)))
        axes[idx].set_yticklabels(words_list)
        axes[idx].set_xlabel('Frequency', fontweight='bold')
        axes[idx].set_title(f'Top Words - {sentiment.capitalize()}', 
                           fontsize=12, fontweight='bold')
        axes[idx].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('outputs/eda/top_words_by_sentiment.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Top words analysis saved\n")


def generate_summary_statistics(df: pd.DataFrame):
    """
    Generate and print summary statistics.
    """
    print("="*60)
    print("DATASET SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\nTotal tweets: {len(df):,}")
    print(f"Airlines covered: {df['airline'].nunique()}")
    print(f"Date range: {df['tweet_created'].min()} to {df['tweet_created'].max()}")
    
    print("\n--- Sentiment Distribution ---")
    for sentiment, count in df['airline_sentiment'].value_counts().items():
        pct = count / len(df) * 100
        print(f"{sentiment.capitalize():12} {count:6,} ({pct:5.2f}%)")
    
    print("\n--- Text Length Statistics ---")
    df['text_length'] = df['text'].str.len()
    print(f"Average length: {df['text_length'].mean():.1f} characters")
    print(f"Median length:  {df['text_length'].median():.1f} characters")
    print(f"Max length:     {df['text_length'].max()} characters")
    print(f"Min length:     {df['text_length'].min()} characters")
    
    print("\n--- Top Airlines by Tweet Volume ---")
    top_airlines = df['airline'].value_counts().head(5)
    for airline, count in top_airlines.items():
        print(f"{airline:20} {count:6,} tweets")
    
    print("\n" + "="*60)


def main():
    """
    Main EDA pipeline - runs all analyses and generates visualizations.
    """
    print("\n" + "="*60)
    print("TWITTER AIRLINE SENTIMENT - EXPLORATORY DATA ANALYSIS")
    print("="*60 + "\n")
    
    # Create output directory
    create_output_dir()
    
    # Load data
    df = load_dataset()
    
    # Generate visualizations
    plot_sentiment_distribution(df)
    plot_text_length_analysis(df)
    generate_wordclouds(df)
    plot_airline_comparison(df)
    plot_top_words(df)
    
    # Print summary
    generate_summary_statistics(df)
    
    print("\n" + "="*60)
    print("EDA COMPLETE!")
    print("All visualizations saved to: outputs/eda/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()