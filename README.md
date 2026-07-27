# 📰 Fake News Detection System Using GRU

A Deep Learning web application built with **Streamlit**, **TensorFlow**, and **Gated Recurrent Units (GRU)** that classifies news content as **REAL** or **FAKE**.

---

## 📌 Features

- **Deep Learning Model:** Uses a trained **Gated Recurrent Unit (GRU)** neural network for Natural Language Processing (NLP).
- **Interactive UI:** Clean and simple web application built using **Streamlit**.
- **Confidence Scoring:** Outputs prediction results along with probability confidence percentages.
- **Robust Text Processing:** Cleans and tokenizes input text dynamically before feeding it into the model.

---

## 📁 Repository Structure

```text
my-fake-news-app/
│
├── app.py                      # Main Streamlit interface application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── gru_fake_news_model.keras   # Saved trained GRU TensorFlow model
└── tokenizer.pkl               # Saved tokenizer object