import re
from pyvi import ViTokenizer
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize as en_tokenize
from nltk.corpus import stopwords
import nltk

# Download punkt cho môi trường Render
nltk.download('punkt', download_dir='/opt/render/nltk_data')
nltk.data.path.append('/opt/render/nltk_data')
nltk.download('stopwords', download_dir='/opt/render/nltk_data')

# Stopwords
stop_words_en = set(stopwords.words("english"))
stop_words_vi = set([w.lower() for w in get_stop_words('vietnamese')])

def preprocess_text(text):
    if not text:
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Detect Vietnamese
    if re.search(r"[ăâđêôơưàáảãạèéẻẽẹìíỉĩịòóỏõọùúủũụỳýỷỹỵ]", text):
        tokens = ViTokenizer.tokenize(text).split()
        tokens = [w for w in tokens if w.lower() not in stop_words_vi]
    else:
        tokens = en_tokenize(text)
        tokens = [w for w in tokens if w.lower() not in stop_words_en]

    return " ".join(tokens)
