import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# UI Title
st.title("📰 Fake News Detector")
st.write("Paste any news article below and I’ll tell you if it’s 🔥 Fake or ✅ Real.")

# Input from user
user_input = st.text_area("Enter news article text here")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess and predict
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        # Output
        if prediction == 0:
            st.error("🚨 This looks like **FAKE** news.")
        else:
            st.success("✅ This looks like **REAL** news.")
