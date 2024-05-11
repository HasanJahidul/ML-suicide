from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_features(corpus, max_features=1000, max_df=0.7, min_df=5):
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df, max_df=max_df)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer
