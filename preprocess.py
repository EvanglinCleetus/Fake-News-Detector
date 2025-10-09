import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\\S+|www\\.\\S+", "", text)
    text = re.sub(r"[^a-z\\s]", " ", text)
    words = [w for w in text.split() if w not in STOPWORDS]
    words = [STEMMER.stem(w) for w in words]
    return " ".join(words)
