import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os


data = pd.read_csv("../data/processed_imdb_debug.csv")
corpus = data["lemmatized_tokens"].apply(lambda tokens: " ".join(eval(tokens))).tolist()
y = data["sentiment"]


X_train, X_test, y_train, y_test = train_test_split(
    corpus, y, test_size=0.2, random_state=42, stratify=y
)


vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


os.makedirs("../models", exist_ok=True)
with open("../models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)


# Logistic Regression

log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
log_model.fit(X_train_tfidf, y_train)
y_pred_log = log_model.predict(X_test_tfidf)

print("=== Logistic Regression (Balanced) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))


with open("../models/logistic_model.pkl", "wb") as f:
    pickle.dump(log_model, f)


#  Random Forest

rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
rf_model.fit(X_train_tfidf, y_train)
y_pred_rf = rf_model.predict(X_test_tfidf)

print("=== Random Forest (Balanced) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


with open("../models/rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

#  SVM

svm_model = SVC(kernel="linear", class_weight="balanced", probability=True)
svm_model.fit(X_train_tfidf, y_train)
y_pred_svm = svm_model.predict(X_test_tfidf)

print("=== SVM (Linear, Balanced) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))


with open("../models/svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)

print("✅ All models and TF-IDF vectorizer saved in '../models/' folder")
