from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureEngineer:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1,2),
            stop_words="english"
        )

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)