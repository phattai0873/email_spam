import re
from underthesea import word_tokenize as vn_tokenize
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
    Xử lý text: lowercase, remove URL/email/number/special chars,
    tokenize, remove stopwords, và trả về text sạch
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)   # remove URL
    text = re.sub(r"\S+@\S+", " ", text)          # remove email
    text = re.sub(r"\d+", " ", text)              # remove numbers
    text = re.sub(r"[^\w\s]", " ", text)          # remove special chars
    text = re.sub(r"\s+", " ", text).strip()      # normalize space

    # Detect Vietnamese by checking dấu tiếng Việt
    if re.search(r"[ăâđêôơưàáảãạèéẻẽẹìíỉĩịòóỏõọùúủũụỳýỷỹỵ]", text):
        tokens = vn_tokenize(text)
        tokens = [w for w in tokens if w not in stop_words_vi]
    else:  # tiếng Anh
        tokens = en_tokenize(text)
        tokens = [w for w in tokens if w.lower() not in stop_words_en]

    return " ".join(tokens)
