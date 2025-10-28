# app.py
import streamlit as st
import joblib
import os

st.set_page_config(page_title="Gmail Spam Detection", page_icon="📧")

st.title("📧 Gmail Spam Detection")
st.write("Paste any email text below and click **Predict** to check if it's Spam or Not Spam.")

MODEL_PATH = "models/spam_classifier.pkl"
VEC_PATH = "models/vectorizer.pkl"

@st.cache_resource
def load_model_and_vectorizer():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
        st.error("❌ Model or vectorizer not found. Please train and save them first.")
        return None, None
    clf = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    return clf, vectorizer

clf, vectorizer = load_model_and_vectorizer()

email_text = st.text_area("✉️ Enter email text here", height=250, placeholder="Type or paste your email content...")

if st.button("Predict"):
    if not email_text.strip():
        st.warning("⚠️ Please enter some text first!")
    else:
        X = vectorizer.transform([email_text])
        pred = clf.predict(X)[0]
        if pred == 1:
            st.error("🚫 Spam Email Detected!")
        else:
            st.success("✅ Not Spam (Safe Email)")
