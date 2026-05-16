# ============================================================
# streamlit_app.py  —  Web Interface using Streamlit
# Run: streamlit run streamlit_app.py
# ============================================================

import streamlit as st
import pickle
from utils.preprocess import clean_text

# --- Page configuration ---
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🔍",
    layout="centered"
)

# --- Load model (cached so it loads only once) ---
@st.cache_resource
def load_model():
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("models/model_name.txt", "r") as f:
        model_name = f.read().strip()
    return model, tfidf, model_name

# --- UI Layout ---
st.title("🔍 Customer Sentiment Analyzer")
st.markdown("Analyze product reviews and classify them as **Positive**, **Negative**, or **Neutral**.")
st.markdown("---")

try:
    model, tfidf, model_name = load_model()
    st.success(f"✅ Model loaded: **{model_name}**")
except Exception:
    st.error("❌ Model not found. Please run `python train_model.py` first.")
    st.stop()

# --- Input area ---
st.subheader("📝 Enter a Product Review")
user_review = st.text_area(
    label="Type or paste a customer review below:",
    placeholder="e.g. This product is amazing! Great quality and fast delivery.",
    height=150
)

# --- Predict button ---
if st.button("🔎 Analyze Sentiment", type="primary"):
    if not user_review.strip():
        st.warning("⚠️ Please enter a review before clicking Analyze.")
    else:
        cleaned = clean_text(user_review)
        features = tfidf.transform([cleaned])
        prediction = model.predict(features)[0]

        # Display result with color and emoji
        if prediction == "positive":
            st.success("😊 Sentiment: **POSITIVE**")
            st.balloons()
        elif prediction == "negative":
            st.error("😞 Sentiment: **NEGATIVE**")
        else:
            st.info("😐 Sentiment: **NEUTRAL**")

        # Show cleaned text (good for viva explanation)
        with st.expander("🔧 See Preprocessed Text"):
            st.code(cleaned)

st.markdown("---")

# --- Sample reviews section ---
st.subheader("📌 Try Sample Reviews")
samples = {
    "👍 Positive Example": "This product is absolutely fantastic! Best purchase I've made. Highly recommend!",
    "👎 Negative Example": "Terrible product. Broke after two days. Complete waste of money. Very disappointed.",
    "😐 Neutral Example":  "It is okay I guess. Nothing special but does the job. Average product overall."
}

for label, text in samples.items():
    if st.button(label):
        cleaned = clean_text(text)
        features = tfidf.transform([cleaned])
        pred = model.predict(features)[0]
        st.markdown(f"**Review:** {text}")
        emoji = {"positive": "😊 POSITIVE", "negative": "😞 NEGATIVE", "neutral": "😐 NEUTRAL"}
        st.markdown(f"**Prediction:** `{emoji.get(pred, pred.upper())}`")

st.markdown("---")
st.caption("Built with Python · Scikit-learn · Streamlit | Academic Semester Project")