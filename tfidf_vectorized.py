from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

df = pd.read_csv("data/stemmed.csv")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["stemmed"])
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
tfidf_df.to_csv("data/tfidf_stemmed.csv", index=False)
