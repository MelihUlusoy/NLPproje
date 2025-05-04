import os
import pandas as pd
import zeyrek

analyzer = zeyrek.MorphAnalyzer()

input_path = "data/temizlenmis.csv"
output_path = "data/lemmatized.csv"

if not os.path.exists(input_path):
    raise FileNotFoundError(f"{input_path} bulunamadı.")

df = pd.read_csv(input_path)

if "stopwordsiz" not in df.columns:
    raise ValueError("'stopwordsiz' sütunu eksik.")

lemmatized_texts = []
for text in df["stopwordsiz"].astype(str):
    lemmas = []
    for word in text.split():
        results = analyzer.analyze(word)
        if results:
            lemma = results[0].lemma
        else:
            lemma = word
        lemmas.append(lemma)
    lemmatized_texts.append(" ".join(lemmas))

df["lemmatized"] = lemmatized_texts
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"✅ Lemmatize edilmiş veri '{output_path}' dosyasına kaydedildi.")
