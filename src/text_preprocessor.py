import re
import nltk
import spacy
from nltk.corpus import stopwords

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

nlp = spacy.load("en_core_web_sm")


class TextPreprocessor:
    def __init__(self, use_lemmatization=False):
        self.use_lemmatization = use_lemmatization

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"http\S+", "", text)  # remove URLs
        text = re.sub(r"<.*?>", "", text)  # remove HTML
        text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove punctuation
        return text

    def tokenize(self, text):
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words]
        return tokens

    def lemmatize(self, tokens):
        doc = nlp(" ".join(tokens))
        return [token.lemma_ for token in doc]

    def preprocess(self, text):
        text = self.clean_text(text)
        tokens = self.tokenize(text)

        if self.use_lemmatization:
            tokens = self.lemmatize(tokens)

        return " ".join(tokens)