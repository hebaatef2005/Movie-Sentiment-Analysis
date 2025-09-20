import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()


print("📂 Loading dataset...")
data = pd.read_csv("../data/IMDB Dataset.csv").head(200)
print("✅ Dataset loaded with shape:", data.shape)


print("🧹 Cleaning text...")
data["cleaned_review"] = data["review"].apply(clean_text)
print("✅ Cleaning done")


print("✂️ Tokenizing...")
data["tokens"] = data["cleaned_review"].apply(word_tokenize)
print("✅ Tokenization done")


print("🛑 Removing stopwords...")
stop_words = set(stopwords.words("english"))
data["filtered_tokens"] = data["tokens"].apply(lambda tokens: [w for w in tokens if w not in stop_words])
print("✅ Stopwords removed")


print("🦾 Lemmatizing...")
lemmatizer = WordNetLemmatizer()
data["lemmatized_tokens"] = data["filtered_tokens"].apply(lambda tokens: [lemmatizer.lemmatize(w) for w in tokens])
print("✅ Lemmatization done")


print("📊 Vectorizing with TF-IDF...")
corpus = data["lemmatized_tokens"].apply(lambda tokens: " ".join(tokens)).tolist()
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(corpus)
print("✅ TF-IDF done. Shape:", X.shape)


data.to_csv("../data/processed_imdb_debug.csv", index=False)
print("💾 Processed IMDB data saved to '../data/processed_imdb_debug.csv'")
