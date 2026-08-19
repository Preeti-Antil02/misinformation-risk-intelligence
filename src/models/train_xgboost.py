# src/models/train_xgboost.py
# run: python -m src.models.train_xgboost

import os
import joblib

from src.data_loader import DataLoader
from src.preprocessing import DataCleaner
from src.features.text_preprocessor import TextPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.features.feature_builder import FeatureBuilder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.sparse import csr_matrix
from scipy.sparse import hstack

from xgboost import XGBClassifier


def train_xgboost(X_train, y_train):

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        use_label_encoder=False,
    )

    param_dist = {
        "max_depth": [3,4,5,6,7],
        "learning_rate": [0.01,0.05,0.1,0.2],
        "n_estimators": [100,200,300],
        "subsample": [0.7,0.8,0.9,1.0]
    }

    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring="f1",
        verbose=2,
        random_state=42,
        n_jobs=1
    )

    search.fit(X_train, y_train)

    print("Best Parameters:", search.best_params_)
    print("Best F1 Score:", search.best_score_)

    return search.best_estimator_


def main():

    # ---------------------------
    # Load data
    # ---------------------------
    print("Loading dataset...")

    loader = DataLoader()
    df = loader.load_combined()

    # ---------------------------
    # Clean data
    # ---------------------------
    print("Cleaning dataset...")

    cleaner = DataCleaner()

    df = cleaner.remove_duplicates(df)
    df = cleaner.remove_nulls(df)
    df = cleaner.remove_short_texts(df)

    # ---------------------------
    # Text preprocessing
    # ---------------------------
    print("Preprocessing text...")

    tp = TextPreprocessor()
    df["text"] = df["text"].apply(tp.basic_clean)
    df["text"] = df["text"].apply(tp.truncate)

    # ---------------------------
    # Split data
    # ---------------------------
    X = df["text"]
    y = df["label"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # ---------------------------
    # TF-IDF features
    # ---------------------------
    print("Building TF-IDF features...")

    fe = FeatureEngineer()

    X_train_tfidf = fe.fit_transform(X_train_text)
    X_test_tfidf = fe.transform(X_test_text)

    # ---------------------------
    # Manipulation features
    # ---------------------------
    print("Building manipulation features...")

    fb = FeatureBuilder()

    train_df = df.loc[X_train_text.index]
    test_df = df.loc[X_test_text.index]

    X_train_numeric = fb.build_features(train_df)
    X_test_numeric = fb.build_features(test_df)

    scaler = StandardScaler()
    X_train_numeric_scaled = scaler.fit_transform(X_train_numeric.values)
    X_test_numeric_scaled = scaler.transform(X_test_numeric.values)

    # ---------------------------
    # Combine features
    # ---------------------------
    print("Combining TF-IDF and manipulation features...")

    X_train_combined = hstack([X_train_tfidf, csr_matrix(X_train_numeric_scaled)])
    X_test_combined  = hstack([X_test_tfidf,  csr_matrix(X_test_numeric_scaled)])

    print("Training shape:", X_train_combined.shape)

    # ---------------------------
    # Train model
    # ---------------------------
    print("Training XGBoost model...")

    model = train_xgboost(X_train_combined, y_train)

    print("XGBoost training complete.")

    # ---------------------------
    # Save models
    # ---------------------------
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/xgboost_model.pkl")
    joblib.dump(fe.vectorizer, "models/tfidf_vectorizer.pkl")
    joblib.dump(scaler, "models/numeric_scaler.pkl")

    print("Model saved: models/xgboost_model.pkl")


if __name__ == "__main__":
    main()