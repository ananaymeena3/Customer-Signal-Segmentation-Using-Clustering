# ============================================================
# utils/preprocess.py
# Text cleaning and dataset loading utilities
# ============================================================

import re
import os
import pandas as pd
import numpy as np


def clean_text(text):
    """
    Cleans a raw review string for ML processing.

    Steps:
      1. Lowercase everything
      2. Remove HTML tags (e.g., <br />)
      3. Keep only alphabetic characters (remove numbers, punctuation)
      4. Collapse multiple spaces into one
      5. Strip leading/trailing whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()                              # 1. lowercase
    text = re.sub(r"<[^>]+>", " ", text)            # 2. remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)           # 3. keep letters only
    text = re.sub(r"\s+", " ", text).strip()        # 4-5. clean whitespace

    return text


def label_from_rating(rating):
    """
    Converts a numeric star rating (1–5) to a binary sentiment label.
      4-5 stars → 'positive'
      1-3 stars → 'negative'
    """
    if rating >= 4:
        return "positive"
    else:
        return "negative"


def load_dataset():
    """
    Loads the sentiment dataset from dataset/amazon.csv.
    """
    # Load the Amazon dataset
    file_path = "dataset/amazon.csv"
    
    if not os.path.exists(file_path):
        # Fallback to synthetic data if file not found (for safety)
        print(f"Warning: {file_path} not found. Using synthetic data.")
        return _load_synthetic_data()

    try:
        # Load CSV (it has an unnamed index column at position 0)
        df = pd.read_csv(file_path)
        
        # Check if required columns exist
        if "reviewText" in df.columns and "overall" in df.columns:
            print(f"      Loading from {file_path}...")
            # Convert rating to sentiment (Binary)
            df["sentiment"] = df["overall"].apply(label_from_rating)
            # Rename reviewText to review
            df = df.rename(columns={"reviewText": "review"})
            # Select only necessary columns and drop rows with missing reviews
            df = df[["review", "sentiment"]].dropna(subset=["review"])
            return df
        else:
            print("Warning: CSV missing 'reviewText' or 'overall' columns. Using synthetic.")
            return _load_synthetic_data()
            
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return _load_synthetic_data()


def _load_synthetic_data():
    """
    Internal helper to provide synthetic data as a fallback.
    """
    reviews = [
        # Positive reviews
        ("This product is absolutely amazing! Best purchase ever.", "positive"),
        ("Great quality, fast delivery. Highly recommended!", "positive"),
        ("Excellent product, works perfectly. Very happy with it.", "positive"),
        ("Wonderful! Exceeded my expectations completely.", "positive"),
        ("Love this item! Will definitely buy again.", "positive"),

        # Negative reviews
        ("Terrible product. Broke after just two days of use.", "negative"),
        ("Very disappointed. Does not work as advertised at all.", "negative"),
        ("Worst purchase I have ever made. Total waste of money.", "negative"),
        ("It was okay, but not what I expected.", "negative"), # Neutral treated as neg
        ("Bad experience. The product smelled weird and broke.", "negative"),
    ]
    
    # Duplicate data for size
    reviews = reviews * 20
    np.random.seed(42)
    np.random.shuffle(reviews)
    
    return pd.DataFrame(reviews, columns=["review", "sentiment"])
