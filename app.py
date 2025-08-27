from flask import Flask, render_template, request
import joblib
from text_preprocess import preprocess_text
from gmail_service import get_latest_emails

app = Flask(__name__)

# Load vectorizer + models
vectorizer = joblib.load("models/vectorizer.joblib")
models = {
    "Naive Bayes": joblib.load("models/naive_bayes.joblib"),
    "Logistic Regression": joblib.load("models/logistic.joblib"),
    "SVM": joblib.load("models/svm.joblib"),
    "Random Forest": joblib.load("models/random_forest.joblib")
}

def predict_text(text, algo="Naive Bayes"):
    text_clean = preprocess_text(text)
    X = vectorizer.transform([text_clean])
    model = models[algo]
    return model.predict(X)[0]

# =========================
# Route nhập text từ user
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    text = ""
    algo = None

    if request.method == "POST":
        algo = request.form["algorithm"]
        text = request.form["message"]
        prediction = predict_text(text, algo)

    return render_template("index.html",
                           algorithms=list(models.keys()),
                           prediction=prediction,
                           text=text,
                           algo=algo)

# =========================
# Route lấy email Gmail và dự đoán
# =========================
@app.route("/gmail")
def gmail_emails():
    emails = get_latest_emails(5)
    results = []

    for e in emails:
        preds = {}
        for name in models.keys():
            preds[name] = predict_text(e["body"], name)

        results.append({
            "subject": e["subject"],
            "sender": e["sender"],
            "date": e["date"],
            "body": e["body"],
            "predictions": preds
        })

    return render_template("gmail.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
