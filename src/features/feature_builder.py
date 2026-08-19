import re
import numpy as np
import pandas as pd
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

    def extract_single_text_features(self, text: str) -> dict:
        """Optimized single-pass feature extraction for low-latency real-time inference."""
        s_text = str(text)
        s_lower = s_text.lower()
        all_words = s_text.split()
        num_words = len(all_words)
        
        # 1. Exclamation count
        excl_cnt = float(s_text.count("!"))
        
        # 2. Capital words & ratio (single regex pass)
        words_caps = re.findall(r'\b[A-Z]{2,}\b', s_text)
        sensational_caps = [w for w in words_caps if w.upper() not in self.COMMON_ACRONYMS]
        cap_cnt = float(len(sensational_caps))
        cap_ratio = float(min(1.0, cap_cnt / max(1, num_words))) if num_words > 0 else 0.0
        
        # 3. Extreme keywords count
        extreme_cnt = float(sum(s_lower.count(word) for word in self.extreme_words))
        
        # 4. Sentiment & subjectivity (single TextBlob pass)
        try:
            blob_sentiment = TextBlob(s_text).sentiment
            pol = float(blob_sentiment.polarity)
            subj = float(blob_sentiment.subjectivity)
        except Exception:
            pol = 0.0
            subj = 0.0
            
        # 5. Avg word length
        if num_words == 0:
            avg_len = 4.5
        else:
            avg_len = sum(len(w) for w in all_words) / num_words
        avg_len_clamped = float(min(10.0, max(3.0, avg_len)))
        
        return {
            "exclamation_count": excl_cnt,
            "capital_word_count": cap_cnt,
            "capital_ratio": cap_ratio,
            "extreme_keyword_count": extreme_cnt,
            "sentiment_polarity": pol,
            "subjectivity": subj,
            "avg_word_length": avg_len_clamped
        }

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds and returns a DataFrame of the 7 linguistic manipulation features.
        """
        rows = [self.extract_single_text_features(t) for t in df["text"]]
        return pd.DataFrame(rows, index=df.index)[self.feature_names]