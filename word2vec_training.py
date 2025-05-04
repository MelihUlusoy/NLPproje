import pandas as pd
from gensim.models import Word2Vec
import os

# Veri dosyasını oku
input_path = "data/stemmed.csv"
if not os.path.exists(input_path):
    raise FileNotFoundError(f"{input_path} bulunamadı. Lütfen önce preprocessing ve stemming adımlarını tamamlayın.")

df = pd.read_csv(input_path)

# "stemmed" sütununu kontrol et
if "stemmed" not in df.columns:
    raise ValueError("'stemmed' sütunu bulunamadı. Lütfen doğru CSV dosyasını kullanın.")

# Her bir stemlenmiş metni (satır) kelime listesine çevir
sentences = [str(text).split() for text in df["stemmed"].astype(str)]

# Word2Vec eğitim parametreleri
window_size = 5   # Tercihe bağlı pencere boyutu
min_count = 1     # Kelime frekansına göre filtreleme (minimum)
sg = 0            # 0: CBOW, 1: Skip-gram (burada CBOW modeli kullanılıyor)

# Model 1: vektör boyutu 300
model_300 = Word2Vec(sentences, vector_size=300, window=window_size, min_count=min_count, sg=sg)
model_300.save("models/word2vec_stemmed_cbow_win5_dim300.model")
print("Word2Vec modeli (vector_size=300) eğitildi ve 'models/word2vec_stemmed_cbow_win5_dim300.model' dosyasına kaydedildi.")

# Model 2: vektör boyutu 1000
model_1000 = Word2Vec(sentences, vector_size=1000, window=window_size, min_count=min_count, sg=sg)
model_1000.save("models/word2vec_stemmed_cbow_win5_dim1000.model")
print("Word2Vec modeli (vector_size=1000) eğitildi ve 'models/word2vec_stemmed_cbow_win5_dim1000.model' dosyasına kaydedildi.")
