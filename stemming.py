import os
import pandas as pd
import nltk
from TurkishStemmer import TurkishStemmer

# NLTK stopwords verisini indir (gerekirse)
nltk.download('stopwords', quiet=True)

# Dosya yolları
input_path = "data/temizlenmis.csv"
output_path = "data/stemmed.csv"

# Dosya mevcut mu kontrol et
if not os.path.exists(input_path):
    print(f"Hata: '{input_path}' dosyası bulunamadı.")
    exit()

# CSV dosyasını oku
df = pd.read_csv(input_path)

# TurkishStemmer başlat
stemmer = TurkishStemmer()

# Stemleme işlemi
if "stopwordsiz" in df.columns:
    df["stemmed"] = df["stopwordsiz"].astype(str).apply(
        lambda x: " ".join(stemmer.stem(w) for w in x.split())
    )
    df.to_csv(output_path, index=False)
    print(f"Stemlenmiş veri '{output_path}' dosyasına kaydedildi.")
else:
    print("Hata: 'stopwordsiz' sütunu bulunamadı. Önce stopwords çıkarma adımını tamamlayın.")
