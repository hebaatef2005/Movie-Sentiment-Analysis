import streamlit as st
import pickle
import os


MODEL_PATH = "../models"

with open(os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

models = {}
for name in ["logistic_model", "rf_model", "svm_model"]:
    with open(os.path.join(MODEL_PATH, f"{name}.pkl"), "rb") as f:
        models[name] = pickle.load(f)


st.set_page_config(page_title="Sentiment Classifier", page_icon="🤖", layout="centered")

st.title("Sentiment Analysis on Movie Reviews")
st.write("This app classifies movie reviews as **Positive** or **Negative** using different machine learning models trained on my dataset.")


user_input = st.text_area("Enter your review here:", "")


model_choice = st.selectbox(
    "Choose a model:",
    ["Logistic Regression", "Random Forest", "SVM"],
)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text before classification.")
    else:
       
        X_input = vectorizer.transform([user_input])

        
        model_map = {
            "Logistic Regression": "logistic_model",
            "Random Forest": "rf_model",
            "SVM": "svm_model",
        }
        model = models[model_map[model_choice]]

       
        prediction = model.predict(X_input)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[0]

        
        st.subheader("Result")
        if prediction == "positive":
            st.success("Positive review")
        else:
            st.error("Negative review")

        if proba is not None:
            st.write("Probabilities:", {label: f"{p:.2f}" for label, p in zip(model.classes_, proba)})
