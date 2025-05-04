import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Veriyi oku
df = pd.read_csv("data/urun_adlari_large_dataset.csv")

# Sütunları göster
print("Mevcut sütunlar:", df.columns.tolist())

# Sütun adlarını normalize et (küçük harf, boşluk yerine alt çizgi)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Ürün adının olduğu sütunu tespit et (otomatik)
urun_adi_kolonu = None
for col in df.columns:
    if "title" in col or ("urun" in col and "ad" in col):
        urun_adi_kolonu = col
        break

if urun_adi_kolonu is None:
    raise KeyError("Beklenen 'ürün_adı' veya 'product_title' benzeri sütun CSV dosyasında bulunamadı.")

print(f"Kullanılan sütun: {urun_adi_kolonu}")

# Temizleme fonksiyonu
def temizle(metin):
    metin = str(metin).lower()
    metin = re.sub(r'[^a-zA-Z0-9çğıöşüÇĞİÖŞÜ\s]', '', metin)
    return metin

df["temiz"] = df[urun_adi_kolonu].astype(str).apply(temizle)

# Stopword çıkar
stop_words = set(stopwords.words("turkish"))
df["stopwordsiz"] = df["temiz"].apply(lambda x: " ".join([w for w in x.split() if w not in stop_words]))

# Kaydet
df.to_csv("data/temizlenmis.csv", index=False)
print("✅ Temizlenmiş veri 'data/temizlenmis.csv' dosyasına kaydedildi.")
