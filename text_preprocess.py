import re
import os
import nltk
from pyvi import ViTokenizer
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize as en_tokenize
from nltk.corpus import stopwords

# Setup NLTK path (Render compatible)
NLTK_PATH = "/opt/render/nltk_data"
os.makedirs(NLTK_PATH, exist_ok=True)
nltk.data.path.append(NLTK_PATH)

# Ensure required NLTK data
for pkg in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.download(pkg, download_dir=NLTK_PATH, quiet=True)
    except Exception as e:
        print(f"NLTK download failed for {pkg}: {e}")

# Stopwords
stop_words_en = set(stopwords.words("english"))
stop_words_vi = set(w.lower() for w in get_stop_words("vietnamese"))

def preprocess_text(text: str) -> str:
    if not text:
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)   # URLs
    text = re.sub(r"\S+@\S+", " ", text)          # emails
    text = re.sub(r"\d+", " ", text)              # numbers
    text = re.sub(r"[^\w\s]", " ", text)          # punctuation
    text = re.sub(r"\s+", " ", text).strip()      # normalize spaces

    # Detect Vietnamese characters
    if re.search(r"[ăâđêôơưàáảãạèéẻẽẹìíỉĩịòóỏõọùúủũụỳýỷỹỵ]", text):
        tokens = ViTokenizer.tokenize(text).split()
        tokens = [w for w in tokens if w not in stop_words_vi]
    else:
        tokens = en_tokenize(text)
        tokens = [w for w in tokens if w not in stop_words_en]

    return " ".join(tokens)
