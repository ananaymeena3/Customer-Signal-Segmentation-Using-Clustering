# 🎓 VIVA QUESTIONS & ANSWERS
## Customer Sentiment Analysis Project

---

### Q1. What is Sentiment Analysis?
**A:** Sentiment Analysis (also called Opinion Mining) is an NLP technique used to identify and classify
the emotional tone of text. In our project, it classifies customer reviews as Positive, Negative,
or Neutral using Machine Learning.

---

### Q2. What is TF-IDF and why did you use it?
**A:** TF-IDF stands for **Term Frequency – Inverse Document Frequency**.
- **TF (Term Frequency):** How often a word appears in a single document.
- **IDF (Inverse Document Frequency):** Penalizes words that appear in many documents (common words
  like "the", "is" get low IDF).
- **Why used:** It converts text to numbers so ML algorithms can process it, while giving more
  importance to meaningful, rare words. It's better than CountVectorizer because it handles common
  words automatically.

---

### Q3. Why not use CountVectorizer instead of TF-IDF?
**A:** CountVectorizer simply counts word occurrences and treats all words equally. TF-IDF is smarter
because it down-weights very common words (stop words) even without explicitly removing them, and
highlights rare but meaningful terms. This leads to better model performance on text data.

---

### Q4. What is Logistic Regression? Why use it for classification?
**A:** Despite its name, Logistic Regression is a **classification** algorithm. It uses the sigmoid
function to output a probability (0 to 1) and classifies based on a threshold (usually 0.5).
It works well for linearly separable data and is fast and interpretable, making it a great
baseline model for text classification.

---

### Q5. Explain Naive Bayes for text classification.
**A:** Naive Bayes is a probabilistic classifier based on **Bayes' Theorem**. The "naive" part means
it assumes all features (words) are independent of each other, which is rarely true in language
but still works surprisingly well in practice. MultinomialNB is specifically designed for
discrete word count features, making it ideal for text data.

---

### Q6. What is a Confusion Matrix?
**A:** A Confusion Matrix is a grid that shows:
- **True Positives (TP):** Correctly predicted positives
- **True Negatives (TN):** Correctly predicted negatives
- **False Positives (FP):** Predicted positive but actually negative (Type I error)
- **False Negatives (FN):** Predicted negative but actually positive (Type II error)

From these values we calculate Accuracy, Precision, Recall, and F1-Score.

---

### Q7. What is the difference between Precision, Recall, and F1-Score?
**A:**
- **Precision = TP / (TP + FP):** Of all predicted positives, how many were actually positive?
- **Recall = TP / (TP + FN):** Of all actual positives, how many did we correctly find?
- **F1-Score = 2 * (Precision * Recall) / (Precision + Recall):** Harmonic mean of both.

F1-Score is used when you need a balance between Precision and Recall, especially with
imbalanced datasets.

---

### Q8. Why did you split data into 80% train and 20% test?
**A:** We need separate data for training (model learns patterns) and testing (model evaluated
on unseen data). If we test on training data, the model appears accurate but fails on new
data — this is called **overfitting**. 80/20 is a standard and effective split ratio.

---

### Q9. What is overfitting and how do you prevent it?
**A:** Overfitting is when a model learns the training data too well, including its noise, and
performs poorly on new data. Prevention methods:
1. Use train-test split / cross-validation
2. Reduce model complexity
3. Add more training data
4. Use regularization (e.g., C parameter in Logistic Regression)

---

### Q10. What preprocessing steps did you apply and why?
**A:**
1. **Lowercase** — "Great" and "great" are the same word
2. **Remove HTML tags** — online reviews often contain HTML
3. **Remove punctuation/numbers** — not meaningful for sentiment
4. **Remove extra spaces** — clean the data format

---

### Q11. What dataset did you use?
**A:** We used a synthetic dataset for demonstration. For a production-quality submission, the
Amazon Product Reviews dataset from Kaggle (bittlingmayer/amazonreviews) is recommended.
It contains millions of real Amazon reviews labeled as positive or negative.

---

### Q12. What are the possible improvements to this project?
**A:**
1. Use **BERT** or **LSTM** deep learning models for better accuracy
2. **Multilingual support** using models like mBERT
3. **Aspect-based sentiment** (analyze price, quality, delivery separately)
4. **Real-time data scraping** from e-commerce websites
5. **Deploy as REST API** using Flask/FastAPI
6. Use **Word2Vec or GloVe** embeddings instead of TF-IDF

---

### Q13. What is SVM and when should you use it?
**A:** SVM (Support Vector Machine) finds the optimal hyperplane that maximizes the margin between
classes. LinearSVC is a fast variant of SVM suitable for high-dimensional text data. SVMs are
particularly good for text classification because text naturally has many features (words) and
SVMs handle high-dimensional spaces well.

---

### Q14. What is the role of the `ngram_range=(1,2)` parameter in TF-IDF?
**A:** This means we include both **unigrams** (single words like "good") and **bigrams** (two-word
phrases like "not good", "very bad"). Bigrams capture context that single words miss. For
example, "not good" should be negative but word-by-word "not" and "good" could be confusing.

---

### Q15. How does Streamlit work in this project?
**A:** Streamlit is a Python library that turns Python scripts into interactive web apps with minimal
code. In our project, it provides a browser-based UI where users can enter a review and
instantly see the sentiment prediction. It runs a local web server on port 8501.

---

*💡 Tip: Practice explaining the pipeline from raw text → preprocessing → TF-IDF → model → prediction. 
That end-to-end explanation is the most commonly asked viva question.*