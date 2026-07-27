import pickle
import numpy as np
import re
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page configuration
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection System")
st.write("Enter a news article or headline below to verify its authenticity using our trained GRU deep learning model.")

# 1. Load Model and Tokenizer
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("gru_fake_news_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_assets()

# 2. Text Preprocessing Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 3. User Input Text Area
user_input = st.text_area("News Article Content:", height=150, placeholder="Paste article text here...")

# 4. Predict Button
if st.button("Analyze News"):
    if user_input.strip() == "":
        st.warning("Please paste or type some news text first.")
    else:
        # Preprocess and sequence
        cleaned = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=200, padding="post", truncating="post")
        
        # Prediction
        prediction_prob = float(model.predict(padded)[0][0])
        
        st.subheader("Prediction Result:")
        if 0.45 <= prediction_prob <= 0.55:
          st.warning(f"⚠️ **UNCERTAIN** (Confidence: {prediction_prob * 100:.2f}%)\n\n*The model needs more article body text to decide accurately.*")
        elif prediction_prob > 0.55:
            st.error(f"🚨 **FAKE NEWS** (Confidence: {prediction_prob * 100:.2f}%)")
        else:
            st.success(f"✅ **REAL NEWS** (Confidence: {(1 - prediction_prob) * 100:.2f}%)")