# ============================================================
# app.py  —  Command-Line Sentiment Predictor
# Run: python app.py
# ============================================================

import pickle
import sys
from utils.preprocess import clean_text

# ----------------------------------------------------------
# Load saved model and vectorizer
# ----------------------------------------------------------
def load_model():
    try:
        with open("models/best_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("models/tfidf_vectorizer.pkl", "rb") as f:
            tfidf = pickle.load(f)
        with open("models/model_name.txt", "r") as f:
            model_name = f.read().strip()
        return model, tfidf, model_name
    except FileNotFoundError:
        print("\n[ERROR] Model not found. Please run train_model.py first.")
        print("        Command: python train_model.py\n")
        sys.exit(1)


# ----------------------------------------------------------
# Predict sentiment for a given review text
# ----------------------------------------------------------
def predict_sentiment(text, model, tfidf):
    cleaned = clean_text(text)
    features = tfidf.transform([cleaned])
    prediction = model.predict(features)[0]

    # Emoji mapping for a friendly output
    emoji_map = {
        "positive": "😊 POSITIVE",
        "negative": "😞 NEGATIVE",
        "neutral":  "😐 NEUTRAL"
    }
    return emoji_map.get(prediction, prediction.upper())


# ----------------------------------------------------------
# Main interactive loop
# ----------------------------------------------------------
def main():
    print("=" * 55)
    print("   CUSTOMER SENTIMENT ANALYSIS — CLI App")
    print("=" * 55)

    model, tfidf, model_name = load_model()
    print(f"\n  Loaded model : {model_name}")
    print("  Type a product review to analyze its sentiment.")
    print("  Type 'quit' or 'exit' to stop.\n")
    print("-" * 55)

    # --- Sample reviews to demo the system ---
    demo_reviews = [
        "This product is absolutely fantastic! Highly recommend.",
        "Complete waste of money. Terrible quality.",
        "It's okay I guess, nothing special really."
    ]

    print("  Running quick demo with sample reviews:\n")
    for rev in demo_reviews:
        result = predict_sentiment(rev, model, tfidf)
        print(f"  Review    : {rev}")
        print(f"  Sentiment : {result}\n")

    print("-" * 55)
    print("  Now enter your own review:\n")

    # --- Interactive loop ---
    while True:
        user_input = input("  Enter review: ").strip()

        if not user_input:
            print("  [!] Please enter some text.\n")
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\n  Exiting. Thank you!\n")
            break

        result = predict_sentiment(user_input, model, tfidf)
        print(f"  Sentiment  : {result}\n")


if __name__ == "__main__":
    main()