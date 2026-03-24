# src/models/train_baseline.py
# run: python -m src.models.train_baseline

import os
import joblib

from src.data_loader import DataLoader
from src.preprocessing import DataCleaner
from src.features.text_preprocessor import TextPreprocessor
from src.features.feature_engineering import FeatureEngineer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def main():

    # -----------------------------
    # 1. Load dataset
    # -----------------------------
    print("Loading dataset...")

    loader = DataLoader()
    df = loader.load_combined()

    # -----------------------------
    # 2. Clean dataset
    # -----------------------------
    print("Cleaning dataset...")

    cleaner = DataCleaner()

    df = cleaner.remove_duplicates(df)
    df = cleaner.remove_nulls(df)
    df = cleaner.remove_short_texts(df)

    # -----------------------------
    # 3. Text preprocessing
    # -----------------------------
    print("Preprocessing text...")

    tp = TextPreprocessor()
    df["text"] = df["text"].apply(tp.basic_clean)
    df["text"] = df["text"].apply(tp.truncate)

    # -----------------------------
    # 4. Split features / labels
    # -----------------------------
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # -----------------------------
    # 5. TF-IDF Feature Extraction
    # -----------------------------
    print("Building TF-IDF features...")

    fe = FeatureEngineer()

    X_train_tfidf = fe.fit_transform(X_train)
    X_test_tfidf = fe.transform(X_test)

    print("TF-IDF shape:", X_train_tfidf.shape)

    # -----------------------------
    # 6. Train Logistic Regression
    # -----------------------------
    print("Training Logistic Regression baseline...")

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train_tfidf, y_train)

    print("Baseline model trained.")

    # -----------------------------
    # 7. Save model + vectorizer
    # -----------------------------
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/baseline_logistic.pkl")
    joblib.dump(fe.vectorizer, "models/tfidf_vectorizer.pkl")

    print("Model saved: models/baseline_logistic.pkl")
    print("Vectorizer saved: models/tfidf_vectorizer.pkl")


if __name__ == "__main__":
    main()