import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from preprocess import clean_text
import os

# ✅ Step 1: Load dataset
df = pd.read_csv("dataset.csv")

# ✅ Step 2: Basic cleaning
df.dropna(inplace=True)
df['text_clean'] = df['text'].apply(clean_text)

# ✅ Step 3: Handle small or imbalanced datasets safely
unique_labels = df['label'].unique()
if len(unique_labels) < 2:
    print("⚠️ Only one label found in dataset — duplicating for safe training.")
    df = pd.concat([df, df])  # Duplicate rows to allow both train/test splits
    df['label'] = df['label'].astype(str)

# ✅ Step 4: Safe train/test split (no stratify)
X_train, X_test, y_train, y_test = train_test_split(
    df['text_clean'], df['label'], test_size=0.2, random_state=42
)

# ✅ Step 5: Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ✅ Step 6: Train the model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# ✅ Step 7: Evaluate accuracy
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully. Accuracy: {accuracy * 100:.2f}%")

# ✅ Step 8: Save model and vectorizer
import os
import joblib

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")
joblib.dump(vectorizer, "models/vectorizer.joblib")

print("✅ Model and vectorizer saved successfully!")

print("🎯 Model and vectorizer saved in 'models/' folder.")

