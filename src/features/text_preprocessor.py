from pydoc import text
import re
import spacy

nlp = spacy.load("en_core_web_sm")


class TextPreprocessor:
    def __init__(self, remove_stopwords=False, lemmatize=False):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize

    def basic_clean(self, text):
        text = text.lower()
        text = re.sub(r"http\S+", "", text)        # Remove URLs
        text = re.sub(r"<.*?>", "", text)         # Remove HTML
        text = re.sub(r"[^a-zA-Z\s]", "", text)   # Remove punctuation
        return text

    def advanced_clean(self, text):
        text = self.basic_clean(text)

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
    
    def truncate(self, text, max_words=150):
        words = text.split()
        if len(words) > max_words:
            return " ".join(words[:max_words])
        return text