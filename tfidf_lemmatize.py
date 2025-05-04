import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Giriş yolu
input_path = "data/lemmatized.csv"

# Dosya kontrolü
if not os.path.exists(input_path):
    raise FileNotFoundError(f"{input_path} bulunamadı.")

# CSV oku
df = pd.read_csv(input_path)

# Sütun kontrolü
if "lemmatized" not in df.columns:
    raise ValueError("'lemmatized' sütunu bulunamadı.")

# TF-IDF vektörleştirme
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["lemmatized"].astype(str))

# Sonuçları kaydet (opsiyonel)
with open("models/tfidf_vectorizer_lemmatized.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("TF-IDF matris oluşturuldu ve 'models/tfidf_vectorizer_lemmatized.pkl' olarak kaydedildi.")
