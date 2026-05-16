# ============================================================
# Customer Sentiment Analysis - Model Training Script
# Author: [Your Name] | College: [Your College Name]
# Subject: Machine Learning | Semester: 6th
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

from utils.preprocess import clean_text, load_dataset

# ----------------------------------------------------------
# STEP 1: Load Dataset
# ----------------------------------------------------------
print("=" * 55)
print("   CUSTOMER SENTIMENT ANALYSIS - TRAINING")
print("=" * 55)

print("\n[1/6] Loading dataset...")
df = load_dataset()
print(f"      Dataset shape: {df.shape}")
print(f"      Columns: {list(df.columns)}")
print(f"\n      Label distribution:")
print(df["sentiment"].value_counts().to_string())

# ----------------------------------------------------------
# STEP 2: Text Preprocessing
# ----------------------------------------------------------
print("\n[2/6] Cleaning and preprocessing text...")
df["clean_review"] = df["review"].apply(clean_text)
print("      Sample cleaned review:")
print(f"      Original : {df['review'].iloc[0][:80]}...")
print(f"      Cleaned  : {df['clean_review'].iloc[0][:80]}...")

# ----------------------------------------------------------
# STEP 3: Feature Extraction using TF-IDF
# ----------------------------------------------------------
print("\n[3/6] Extracting features using TF-IDF Vectorizer...")
tfidf = TfidfVectorizer(
    max_features=5000,   # Use top 5000 words
    ngram_range=(1, 2),  # Unigrams + Bigrams
    stop_words="english"
)
X = tfidf.fit_transform(df["clean_review"])
y = df["sentiment"]

print(f"      Feature matrix shape: {X.shape}")

# ----------------------------------------------------------
# STEP 4: Train-Test Split
# ----------------------------------------------------------
print("\n[4/6] Splitting data into Train (80%) and Test (20%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Train samples: {X_train.shape[0]}")
print(f"      Test samples : {X_test.shape[0]}")

# ----------------------------------------------------------
# STEP 5: Train Multiple Models and Compare
# ----------------------------------------------------------
print("\n[5/6] Training models...\n")

models = {
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "Naive Bayes":          MultinomialNB(),
    "SVM (LinearSVC)":      LinearSVC(random_state=42)
}

results = {}
best_acc = 0
best_model_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {"model": model, "accuracy": acc, "y_pred": y_pred}
    print(f"  {name}")
    print(f"    Accuracy : {acc * 100:.2f}%")
    print(f"    Report   :\n{classification_report(y_test, y_pred, zero_division=0)}")

    if acc > best_acc:
        best_acc = acc
        best_model_name = name

print(f"\n  ★ Best Model: {best_model_name} ({best_acc*100:.2f}% accuracy)")

# ----------------------------------------------------------
# STEP 6: Save Best Model + Vectorizer
# ----------------------------------------------------------
print("\n[6/6] Saving best model and vectorizer...")
os.makedirs("models", exist_ok=True)
best_model = results[best_model_name]["model"]

with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("models/model_name.txt", "w") as f:
    f.write(best_model_name)

print(f"      Saved: models/best_model.pkl")
print(f"      Saved: models/tfidf_vectorizer.pkl")

# ----------------------------------------------------------
# PLOTS - Confusion Matrix & Accuracy Bar Chart
# ----------------------------------------------------------
print("\n  Generating evaluation plots...")
os.makedirs("plots", exist_ok=True)

# -- Plot 1: Confusion Matrix for best model
y_pred_best = results[best_model_name]["y_pred"]
cm = confusion_matrix(y_test, y_pred_best, labels=best_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix\n{best_model_name}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png", dpi=150)
plt.close()

# -- Plot 2: Model Accuracy Comparison Bar Chart
acc_names  = list(results.keys())
acc_values = [results[n]["accuracy"] * 100 for n in acc_names]

colors = ["#4CAF50" if n == best_model_name else "#90CAF9" for n in acc_names]
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(acc_names, acc_values, color=colors, edgecolor="white", linewidth=0.7)
ax.set_ylim(0, 110)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_title("Model Accuracy Comparison", fontsize=13, fontweight="bold")
for bar, val in zip(bars, acc_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.xticks(fontsize=9)
plt.tight_layout()
plt.savefig("plots/model_comparison.png", dpi=150)
plt.close()

# -- Plot 3: Sentiment Distribution (Pie Chart)
sentiment_counts = df["sentiment"].value_counts()
colors_pie = ["#66BB6A", "#EF5350", "#42A5F5"]
fig, ax = plt.subplots(figsize=(5, 5))
ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%",
       colors=colors_pie[:len(sentiment_counts)], startangle=140,
       textprops={"fontsize": 11})
ax.set_title("Sentiment Distribution in Dataset", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/sentiment_distribution.png", dpi=150)
plt.close()

print("      Saved: plots/confusion_matrix.png")
print("      Saved: plots/model_comparison.png")
print("      Saved: plots/sentiment_distribution.png")

print("\n" + "=" * 55)
print("   TRAINING COMPLETE!")
print(f"   Best Model : {best_model_name}")
print(f"   Accuracy   : {best_acc * 100:.2f}%")
print("   Run app.py for interactive prediction.")
print("=" * 55 + "\n")