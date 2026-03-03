import re
import spacy

nlp = spacy.load("en_core_web_sm")


class TextPreprocessor:
    def __init__(self, remove_stopwords=False, lemmatize=False):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"http\S+", "", text)        # Remove URLs
        text = re.sub(r"<.*?>", "", text)         # Remove HTML
        text = re.sub(r"[^a-zA-Z\s]", "", text)   # Remove punctuation
        return text

    def preprocess(self, text):
        text = self.clean_text(text)

        doc = nlp(text)

        tokens = []
        for token in doc:
            if token.is_space:
                continue

            if self.remove_stopwords and token.is_stop:
                continue

            if self.lemmatize:
                tokens.append(token.lemma_)
            else:
                tokens.append(token.text)

        return " ".join(tokens)