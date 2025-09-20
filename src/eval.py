import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv("../data/processed_imdb_debug.csv")
corpus = data["lemmatized_tokens"].apply(lambda tokens: " ".join(eval(tokens))).tolist()
y = data["sentiment"]


with open("../models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("../models/logistic_model.pkl", "rb") as f:
    log_model = pickle.load(f)

with open("../models/rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("../models/svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)


X_tfidf = vectorizer.transform(corpus)


models = {
    "Logistic Regression": log_model,
    "Random Forest": rf_model,
    "SVM": svm_model
}

for name, model in models.items():
    y_pred = model.predict(X_tfidf)
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred, labels=["negative", "positive"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["negative", "positive"], yticklabels=["negative", "positive"])
    plt.title(f"Confusion Matrix: {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()
