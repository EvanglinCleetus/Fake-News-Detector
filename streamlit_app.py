import streamlit as st
import joblib
from preprocess import clean_text  # make sure you have preprocess.py in your project

# ✅ Load model and vectorizer
try:
    model = joblib.load("models/model.joblib")
    vectorizer = joblib.load("models/vectorizer.joblib")
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

# ✅ Streamlit UI
st.title("📰 Fake News Detector")

user_input = st.text_area("Enter news text to verify:", height=150)

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Clean the text
        clean = clean_text(user_input)

        # Convert text to vector
        vec = vectorizer.transform([clean])

        # Predict
        pred = model.predict(vec)[0]

        # Display result
        if pred == "FAKE":
            st.error("🚨 This news seems **FAKE**!")
        else:
            st.success("✅ This news seems **REAL**.")
