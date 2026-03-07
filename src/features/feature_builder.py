import re
import numpy as np
import pandas as pd
from textblob import TextBlob


class FeatureBuilder:

    def __init__(self):
        self.extreme_words = ["100%", "guaranteed", "shocking", "secret", "cure"]

    def exclamation_count(self, text):
        return text.count("!")

    def capital_word_count(self, text):
        words = text.split()
        return sum(1 for w in words if w.isupper())

    def capital_ratio(self, text):
        words = text.split()
        if len(words) == 0:
            return 0
        return sum(1 for w in words if w.isupper()) / len(words)

    def extreme_keyword_count(self, text):
        text_lower = text.lower()
        return sum(text_lower.count(word) for word in self.extreme_words)

    def sentiment_polarity(self, text):
        return TextBlob(text).sentiment.polarity

    def subjectivity(self, text):
        return TextBlob(text).sentiment.subjectivity

    def text_length(self, text):
        return len(text.split())

    def avg_word_length(self, text):
        words = text.split()
        if len(words) == 0:
            return 0
        return np.mean([len(w) for w in words])

    def build_features(self, df):

        features = pd.DataFrame()

        features["exclamation_count"] = df["text"].apply(self.exclamation_count)
        features["capital_word_count"] = df["text"].apply(self.capital_word_count)
        features["capital_ratio"] = df["text"].apply(self.capital_ratio)
        features["extreme_keyword_count"] = df["text"].apply(self.extreme_keyword_count)
        features["sentiment_polarity"] = df["text"].apply(self.sentiment_polarity)
        features["subjectivity"] = df["text"].apply(self.subjectivity)
        features["text_length"] = df["text"].apply(self.text_length)
        features["avg_word_length"] = df["text"].apply(self.avg_word_length)

        return features