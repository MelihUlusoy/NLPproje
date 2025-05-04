import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os

# Dosya yolları
stemmed_path = "data/stemmed.csv"
lemmatized_path = "data/lemmatized.csv"

# Dosyaları kontrol et
if not os.path.exists(stemmed_path):
    raise FileNotFoundError(f"{stemmed_path} bulunamadı.")
if not os.path.exists(lemmatized_path):
    raise FileNotFoundError(f"{lemmatized_path} bulunamadı.")

# Verileri oku
df_stem = pd.read_csv(stemmed_path)
df_lemma = pd.read_csv(lemmatized_path)

# Sütun kontrolü
if "stemmed" not in df_stem.columns:
    raise ValueError("'stemmed' sütunu eksik.")
if "lemmatized" not in df_lemma.columns:
    raise ValueError("'lemmatized' sütunu eksik.")

# Frekans hesapla
def get_freq(data_column):
    text = " ".join(data_column.astype(str))
    freq = Counter(text.split())
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    ranks = range(1, len(sorted_freq) + 1)
    frequencies = [f[1] for f in sorted_freq]
    return ranks, frequencies

ranks_stem, freq_stem = get_freq(df_stem["stemmed"])
ranks_lemma, freq_lemma = get_freq(df_lemma["lemmatized"])

# Grafiği çiz
plt.figure(figsize=(12, 7))
plt.loglog(ranks_stem, freq_stem, label="Stemming", color="blue")
plt.loglog(ranks_lemma, freq_lemma, label="Lemmatization", color="green")
plt.xlabel("Kelime Sırası (Rank)")
plt.ylabel("Frekans")
plt.title("Zipf Analizi: Stemming vs Lemmatization")
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.savefig("zipf_stem_vs_lemma.jpg")
plt.show()

print("✅ Zipf karşılaştırma grafiği oluşturuldu ve 'zipf_stem_vs_lemma.jpg' olarak kaydedildi.")
