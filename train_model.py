import os
import re
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from text_preprocess import preprocess_text

# =========================
# 1. Load dataset
# =========================
df1 = pd.read_csv("vi_dataset.csv")
df2 = pd.read_csv("spam.csv")

df1 = df1.rename(columns={"texts_vi": "text", "labels": "label"})
df2 = df2.rename(columns={"Message": "text", "Category": "label"})

df1 = df1[["text", "label"]]
df2 = df2[["text", "label"]]
df = pd.concat([df1, df2], ignore_index=True)

print(df.head())
print(df["label"].value_counts())


df["text_clean"] = df["text"].apply(preprocess_text)

# =========================
# 3. Vectorization
# =========================
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    max_df=0.85,
    min_df=2
)

X_vec = vectorizer.fit_transform(df["text_clean"])
y = df["label"]

# =========================
# 4. Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 5. Đánh giá mô hình + Lưu kết quả
# =========================
results = {}

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="spam", zero_division=1)
    rec = recall_score(y_test, y_pred, pos_label="spam", zero_division=1)
    f1 = f1_score(y_test, y_pred, pos_label="spam", zero_division=1)

    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, zero_division=1, digits=4))

# ----- Naive Bayes -----
param_nb = {"alpha": [0.1, 0.3, 0.5, 1.0]}
grid_nb = GridSearchCV(MultinomialNB(), param_grid=param_nb,
                       cv=5, scoring="f1_macro", n_jobs=-1)
grid_nb.fit(X_train, y_train)
best_nb = grid_nb.best_estimator_
print("\n✅ Naive Bayes Best Params:", grid_nb.best_params_)
evaluate_model("Naive Bayes", best_nb, X_test, y_test)

# ----- Logistic Regression -----
param_log = {
    "C": [0.1, 0.5, 1.0, 2.0],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"]
}
grid_log = GridSearchCV(LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42),
                        param_grid=param_log,
                        cv=5, scoring="f1_macro", n_jobs=-1)
grid_log.fit(X_train, y_train)
best_log = grid_log.best_estimator_
print("\n✅ Logistic Regression Best Params:", grid_log.best_params_)
evaluate_model("Logistic Regression", best_log, X_test, y_test)

# ----- SVM -----
param_svm = {"C": [0.5, 1.0, 1.5, 2.0], "loss": ["squared_hinge"]}
grid_svm = GridSearchCV(LinearSVC(class_weight="balanced", max_iter=5000, random_state=42),
                        param_grid=param_svm,
                        cv=5, scoring="f1_macro", n_jobs=-1)
grid_svm.fit(X_train, y_train)
best_svm = grid_svm.best_estimator_
print("\n✅ SVM Best Params:", grid_svm.best_params_)
evaluate_model("SVM", best_svm, X_test, y_test)

# =========================
# 6. Vẽ biểu đồ
# =========================
metrics = ["Accuracy", "Precision", "Recall", "F1"]
models = list(results.keys())

x = np.arange(len(models))
width = 0.2

plt.figure(figsize=(10, 6))

for i, metric in enumerate(metrics):
    scores = [results[m][metric] for m in models]
    plt.bar(x + i*width, scores, width, label=metric)

plt.xticks(x + width * 1.5, models)
plt.ylabel("Score")
plt.title("So sánh mô hình (Accuracy, Precision, Recall, F1)")
plt.legend()
plt.ylim(0, 1.05)

# Lưu biểu đồ
os.makedirs("results", exist_ok=True)
chart_path = "results/model_comparison.png"
plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"\n✅ Biểu đồ đã lưu tại: {chart_path}")

df_results = pd.DataFrame(results).T
print("\n=== Bảng kết quả đánh giá ===")
print(df_results.round(4))

results_path = "results/evaluation_metrics.csv"
df_results.to_csv(results_path, index=True)
print(f"✅ Kết quả đã lưu tại: {results_path}")

# =========================
# 7. Save models
# =========================
os.makedirs("models", exist_ok=True)
joblib.dump(vectorizer, "models/vectorizer.joblib")
joblib.dump(best_nb, "models/naive_bayes.joblib")
joblib.dump(best_log, "models/logistic.joblib")
joblib.dump(best_svm, "models/svm.joblib")

print("\nSaved best models to /models/")
