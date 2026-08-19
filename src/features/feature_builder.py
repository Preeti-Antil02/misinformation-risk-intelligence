import re
import numpy as np
import pandas as pd
<<<<<<< HEAD
from textblob import TextBlob


class FeatureBuilder:
    """
    Extracts linguistic manipulation, sentiment, and structural signals
    from input news text to augment semantic TF-IDF representations.
    Accurately excludes legitimate institutional acronyms from all-caps counts.
    """

    COMMON_ACRONYMS = {
        "WHO", "US", "UN", "EU", "FDA", "CDC", "ECB", "IMF",
        "NASA", "AI", "UK", "G7", "G20", "PM", "CEO", "COVID",
        "COVID-19", "MRNA", "RSV", "ICJ", "OECD", "NATO", "BBC"
    }

    def __init__(self):
        self.extreme_words = [
            "guaranteed", "shocking", "secret", "cure",
            "bombshell", "exposed", "hoax", "conspiracy",
            "urgent", "unbelievable", "mind-control"
        ]
        self.feature_names = [
            "exclamation_count",
            "capital_word_count",
            "capital_ratio",
            "extreme_keyword_count",
            "sentiment_polarity",
            "subjectivity",
            "avg_word_length"
        ]

    def exclamation_count(self, text: str) -> float:
        """Counts exclamation marks in text."""
        return float(str(text).count("!"))

    def capital_word_count(self, text: str) -> float:
        """Counts truly sensationalized fully capitalized words (excluding legitimate acronyms)."""
        words = re.findall(r'\b[A-Z]{2,}\b', str(text))
        sensational_caps = [w for w in words if w.upper() not in self.COMMON_ACRONYMS]
        return float(len(sensational_caps))

    def capital_ratio(self, text: str) -> float:
        """Calculates ratio of sensational capitalized words to total words."""
        all_words = str(text).split()
        if len(all_words) == 0:
            return 0.0
        words = re.findall(r'\b[A-Z]{2,}\b', str(text))
        sensational_caps = [w for w in words if w.upper() not in self.COMMON_ACRONYMS]
        return float(min(1.0, len(sensational_caps) / max(1, len(all_words))))

    def extreme_keyword_count(self, text: str) -> float:
        """Counts occurrence of sensationalist and conspiracy keywords."""
        text_lower = str(text).lower()
        return float(sum(text_lower.count(word) for word in self.extreme_words))

    def sentiment_polarity(self, text: str) -> float:
        """Computes sentiment polarity in [-1.0, 1.0] using TextBlob."""
        try:
            return float(TextBlob(str(text)).sentiment.polarity)
        except Exception:
            return 0.0

    def subjectivity(self, text: str) -> float:
        """Computes subjectivity score in [0.0, 1.0] using TextBlob."""
        try:
            return float(TextBlob(str(text)).sentiment.subjectivity)
        except Exception:
            return 0.0

    def avg_word_length(self, text: str) -> float:
        """Calculates average word length in characters (clamped to realistic ranges)."""
        words = str(text).split()
        if len(words) == 0:
            return 4.5
        avg_len = sum(len(w) for w in words) / len(words)
        return float(min(10.0, max(3.0, avg_len)))

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds and returns a DataFrame of the 7 linguistic manipulation features.
        """
        features_df = pd.DataFrame(index=df.index)

        features_df["exclamation_count"] = df["text"].apply(self.exclamation_count)
        features_df["capital_word_count"] = df["text"].apply(self.capital_word_count)
        features_df["capital_ratio"] = df["text"].apply(self.capital_ratio)
        features_df["extreme_keyword_count"] = df["text"].apply(self.extreme_keyword_count)
        features_df["sentiment_polarity"] = df["text"].apply(self.sentiment_polarity)
        features_df["subjectivity"] = df["text"].apply(self.subjectivity)
        features_df["avg_word_length"] = df["text"].apply(self.avg_word_length)

        return features_df[self.feature_names]
=======
from streamlit import text
from textblob import TextBlob

from src import features


class FeatureBuilder:

    def __init__(self):
        self.extreme_words = [
        "guaranteed", "shocking", "secret", "cure",
        "bombshell", "exposed", "hoax", "conspiracy",
        "urgent", "breaking", "unbelievable", "mainstream"
        ] 
        self.feature_names = [
        "exclamation_count",
        "capital_word_count",
        "capital_ratio",
        "extreme_keyword_count",
        "sentiment_polarity",
        "subjectivity",
        "avg_word_length"
        ]
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
            return 0.0
        return float(sum(len(w) for w in words) / len(words)) 
    
    def build_features(self, df):

        features = pd.DataFrame()

        features["exclamation_count"] = df["text"].apply(self.exclamation_count)
        features["capital_word_count"] = df["text"].apply(self.capital_word_count)
        features["capital_ratio"] = df["text"].apply(self.capital_ratio)
        features["extreme_keyword_count"] = df["text"].apply(self.extreme_keyword_count)
        features["sentiment_polarity"] = df["text"].apply(self.sentiment_polarity)
        features["subjectivity"] = df["text"].apply(self.subjectivity)
        features["avg_word_length"] = df["text"].apply(self.avg_word_length)

        return features[self.feature_names]
>>>>>>> origin/main
