import streamlit as st
import joblib
import pandas as pd
from preprocess import clean_text

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector using AI")
st.markdown("This app detects if a news article is **Fake or Real** using machine learning.")

try:
    model = joblib.load("models/model.joblib")
except:
    st.error("⚠️ Model not found! Please train it using train_model.py first.")
    st.stop()

st.header("🔍 Check Single News Article")
text = st.text_area("Enter news content here...", height=200)

if st.button("Predict"):
    if text.strip():
        clean = clean_text(text)
        pred = model.predict([clean])[0]
        label = "✅ REAL NEWS" if pred == 1 else "🚫 FAKE NEWS"
        st.subheader(f"Prediction: {label}")
    else:
        st.warning("Please enter some text.")

st.markdown("---")
st.header("📂 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file with a column named 'text'", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "text" not in df.columns:
        st.error("CSV must have a 'text' column.")
    else:
        df['clean_text'] = df['text'].astype(str).apply(clean_text)
        df['prediction'] = model.predict(df['clean_text'])
        df['prediction_label'] = df['prediction'].apply(lambda x: "REAL" if x == 1 else "FAKE")
        st.write("✅ Predictions:")
        st.dataframe(df[['text', 'prediction_label']].head(20))
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")
