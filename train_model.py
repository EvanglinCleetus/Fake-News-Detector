import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib
import os
from preprocess import clean_text

df = pd.read_csv("dataset.csv")
df['text_clean'] = df['text'].apply(clean_text)
df['label'] = df['label'].map(lambda x: 1 if str(x).upper() == 'REAL' else 0)

X_train, X_test, y_train, y_test = train_test_split(
    df['text_clean'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('lr', LogisticRegression(max_iter=1000))
])

print("Training model...")
pipe.fit(X_train, y_train)

preds = pipe.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Accuracy: {acc:.3f}")
print("Classification Report:")
print(classification_report(y_test, preds, target_names=['FAKE', 'REAL']))

os.makedirs("models", exist_ok=True)
joblib.dump(pipe, "models/model.joblib")
print("✅ Model saved to models/model.joblib")
