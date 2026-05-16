# ============================================================
# utils/preprocess.py
# Text cleaning and dataset loading utilities
# ============================================================

import re
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
    Converts a numeric star rating (1–5) to a sentiment label.
      4-5 stars → 'positive'
      3   stars → 'neutral'
      1-2 stars → 'negative'
    """
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"


def load_dataset():
    """
    Loads the sentiment dataset.

    HOW TO USE A REAL DATASET (recommended for submission):
    -------------------------------------------------------
    Option A - Amazon Reviews (Kaggle):
      1. Download from:
         https://www.kaggle.com/datasets/bittlingmayer/amazonreviews
      2. Place 'train.ft.txt' in the data/ folder
      3. Uncomment the Amazon loader block below

    Option B - IMDB Movie Reviews (built-in with sklearn):
      Uses sklearn's fetch_20newsgroups -- no download needed
      (already works out of the box for demo)

    Option C - Use the included synthetic sample dataset (default)
      Works immediately with no downloads
    -------------------------------------------------------
    """

    # ---- Option C: Synthetic sample dataset (default, works immediately) ----
    reviews = [
        # Positive reviews
        ("This product is absolutely amazing! Best purchase ever.", "positive"),
        ("Great quality, fast delivery. Highly recommended!", "positive"),
        ("Excellent product, works perfectly. Very happy with it.", "positive"),
        ("Wonderful! Exceeded my expectations completely.", "positive"),
        ("Love this item! Will definitely buy again.", "positive"),
        ("Super useful and easy to use. Great value for money.", "positive"),
        ("Outstanding quality. I am completely satisfied.", "positive"),
        ("Perfect! Exactly as described. Five stars!", "positive"),
        ("Really good product. Fast shipping and well packed.", "positive"),
        ("Fantastic purchase. Works great, looks great too.", "positive"),
        ("Brilliant product! Arrived on time and works perfectly.", "positive"),
        ("Very satisfied. Great product for the price.", "positive"),
        ("Superb quality. Packaging was excellent too.", "positive"),
        ("Incredible product! Highly recommend to everyone.", "positive"),
        ("Nice product. Good build quality and solid feel.", "positive"),

        # Negative reviews
        ("Terrible product. Broke after just two days of use.", "negative"),
        ("Very disappointed. Does not work as advertised at all.", "negative"),
        ("Worst purchase I have ever made. Total waste of money.", "negative"),
        ("Poor quality. Stopped working within a week.", "negative"),
        ("Do not buy this. It is complete garbage.", "negative"),
        ("Extremely bad experience. Product arrived damaged.", "negative"),
        ("Horrible product. Returned it immediately.", "negative"),
        ("Defective item. Customer service was unhelpful.", "negative"),
        ("Not worth the money at all. Very cheap quality.", "negative"),
        ("Awful product. Nothing like the pictures shown.", "negative"),
        ("Bad experience. The product smelled weird and broke.", "negative"),
        ("Waste of time and money. Very poor build.", "negative"),
        ("Disgusting quality. Fell apart in my hands.", "negative"),
        ("Never buying from here again. Terrible experience.", "negative"),
        ("Frustrated and disappointed. Product is unusable.", "negative"),

        # Neutral reviews
        ("It is okay, nothing special. Does the job I suppose.", "neutral"),
        ("Average product. Nothing great but nothing terrible.", "neutral"),
        ("Decent enough for the price. Some good, some bad.", "neutral"),
        ("Works as expected. Not amazing, not terrible.", "neutral"),
        ("Mediocre quality. Product is just okay overall.", "neutral"),
        ("It is fine. Would probably not buy again though.", "neutral"),
        ("So-so product. Has its pros and cons honestly.", "neutral"),
        ("Neither good nor bad. It just exists as a product.", "neutral"),
        ("Passable item. Could be better but not the worst.", "neutral"),
        ("Regular product. Meets basic needs but nothing more.", "neutral"),
    ]

    # Duplicate data to simulate a bigger dataset (for demo purposes)
    reviews = reviews * 6   # 40 reviews → 240 samples

    # Shuffle for randomness
    np.random.seed(42)
    np.random.shuffle(reviews)

    df = pd.DataFrame(reviews, columns=["review", "sentiment"])
    return df

    # ---- Option A: Amazon Reviews (uncomment if you have the file) ----
    # lines = open("data/train.ft.txt", encoding="utf-8").readlines()[:5000]
    # records = []
    # for line in lines:
    #     label = "positive" if line.startswith("__label__2") else "negative"
    #     text = line.split(" ", 1)[1].strip()
    #     records.append((text, label))
    # return pd.DataFrame(records, columns=["review", "sentiment"])

    # ---- Option B: Load from a CSV file ----
    # df = pd.read_csv("data/reviews.csv")
    # df = df.rename(columns={"text": "review", "label": "sentiment"})
    # return df