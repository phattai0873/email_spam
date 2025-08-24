# text_preprocess.py

import re
from pyvi import ViTokenizer  # tokenize tiếng Việt
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize as en_tokenize
from nltk.corpus import stopwords
import nltk

# =========================
# Chuẩn bị stopwords
# =========================
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

stop_words_en = set(stopwords.words("english"))
vn_stopwords = get_stop_words('vietnamese')
stop_words_vi = set(vn_stopwords)

# =========================
# Hàm tiền xử lý text
# =========================
def preprocess_text(text):
    """
    Xử lý text:
    - Chuyển về lowercase
    - Xóa URL, email, số, ký tự đặc biệt
    - Tokenize (tiếng Việt dùng Pyvi, tiếng Anh dùng NLTK)
    - Loại stopwords
    - Trả về text sạch
    """
    if not text:
        return ""

    # Lowercase
    text = str(text).lower()

    # Xóa URL, email, số, ký tự đặc biệt
    text = re.sub(r"http\S+|www\S+", " ", text)   # remove URL
    text = re.sub(r"\S+@\S+", " ", text)          # remove email
    text = re.sub(r"\d+", " ", text)              # remove numbers
    text = re.sub(r"[^\w\s]", " ", text)          # remove special chars
    text = re.sub(r"\s+", " ", text).strip()      # normalize spaces

    # Detect Vietnamese bằng dấu tiếng Việt
    if re.search(r"[ăâđêôơưàáảãạèéẻẽẹìíỉĩịòóỏõọùúủũụỳýỷỹỵ]", text):
        # Tokenize tiếng Việt
        tokens = ViTokenizer.tokenize(text).split()
        # Loại stopwords
        tokens = [w for w in tokens if w not in stop_words_vi]
    else:
        # Tokenize tiếng Anh
        tokens = en_tokenize(text)
        tokens = [w for w in tokens if w.lower() not in stop_words_en]

    return " ".join(tokens)
