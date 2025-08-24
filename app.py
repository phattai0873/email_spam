from flask import Flask, render_template, request
import joblib
from gmail_service import get_latest_emails

app = Flask(__name__)

# Load vectorizer và models
vectorizer = joblib.load("models/vectorizer.joblib")
models = {
    "Naive Bayes": joblib.load("models/naive_bayes.joblib"),
    "Logistic Regression": joblib.load("models/logistic.joblib"),
    "SVM": joblib.load("models/svm.joblib")
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    text = ""
    algo = None

    if request.method == "POST":
        algo = request.form["algorithm"]
        text = request.form["message"]

        # vector hóa văn bản nhập vào
        X = vectorizer.transform([text])

        # lấy model đã load sẵn
        model = models[algo]
        prediction = model.predict(X)[0]

    return render_template("index.html",
                           algorithms=list(models.keys()),
                           prediction=prediction,
                           text=text,
                           algo=algo)

@app.route("/gmail")
def gmail_emails():
    emails = get_latest_emails(5)
    results = []

    for e in emails:
        preds = {}
        for name, model in models.items():
            vec = vectorizer.transform([e["body"]])
            preds[name] = model.predict(vec)[0]

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
