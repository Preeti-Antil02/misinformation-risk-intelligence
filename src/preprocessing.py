import pandas as pd


class DataCleaner:
    def __init__(self, min_words=3):
        self.min_words = min_words

    def remove_duplicates(self, df):
        return df.drop_duplicates(subset="text")

    def remove_nulls(self, df):
        return df.dropna(subset=["text"])

    def remove_short_texts(self, df):
        return df[df["text"].str.split().str.len() >= self.min_words]

    def clean(self, df):
        df = self.remove_nulls(df)
        df = self.remove_duplicates(df)
        df = self.remove_short_texts(df)
        df = df.reset_index(drop=True)
        return df