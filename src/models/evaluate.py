# src/models/evaluate.py
# run: python -m src.models.evaluate

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import DataLoader
from src.preprocessing import DataCleaner
from src.features.text_preprocessor import TextPreprocessor
from src.features.feature_builder import FeatureBuilder
from src.models.bert_model import BertClassifier
from src.risk_scoring import RiskScorer
from src.explainability.shap_explainer import XGBExplainer

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from scipy.sparse import hstack


def evaluate_model(name, model, X_test, y_test, risk_scorer, threshold=0.5):

    y_prob = model.predict_proba(X_test)

    if len(y_prob.shape) > 1:
        y_prob = y_prob[:, 1]

    y_pred = (y_prob >= threshold).astype(int)

    risk_labels = risk_scorer.score_batch(y_prob)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    print(f"\n===== {name} =====")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1-score:", f1)
    print("ROC-AUC:", roc)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nSample Risk Scores:")
    for i in range(10):
        print(f"Prob: {y_prob[i]:.4f} → Risk: {risk_labels[i]}")

    print("\nRisk Distribution:")
    print(pd.Series(risk_labels).value_counts())

    return y_prob


def main():

    # ---------------------------
    # Load dataset
    # ---------------------------
    loader = DataLoader()
    df = loader.load_combined()

    cleaner = DataCleaner()
    df = cleaner.remove_duplicates(df)
    df = cleaner.remove_nulls(df)
    df = cleaner.remove_short_texts(df)

    # Save raw text BEFORE any cleaning — used for BERT + numeric features
    df["raw_text"] = df["text"].copy()

    # Clean text for TF-IDF only
    tp = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    df["text"] = df["text"].apply(tp.advanced_clean)

    X = df["text"]
    y = df["label"]
    print("\nLabel distribution:")
    print(y.value_counts())

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Raw text for BERT — same indices as X_test_text
    X_test_raw = df.loc[X_test_text.index, "raw_text"]  # ← FIX: raw text for BERT

    # ---------------------------
    # Load saved models
    # ---------------------------
    lr_baseline = joblib.load("models/baseline_logistic.pkl")
    xgb_model = joblib.load("models/xgboost_model.pkl")

    print("\nLoading fine-tuned BERT model...")
    bert_classifier = BertClassifier()
    bert_classifier.load("models/bert_finetuned")

    tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    scaler = joblib.load("models/numeric_scaler.pkl")

    # ---------------------------
    # TF-IDF Features
    # ---------------------------
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

    # ---------------------------
    # Manipulation Features
    # ---------------------------
    fb = FeatureBuilder()

    test_df_raw = df.loc[X_test_text.index].copy()
    test_df_raw["text"] = test_df_raw["raw_text"]
    X_test_numeric = fb.build_features(test_df_raw)
    X_test_numeric = scaler.transform(X_test_numeric)

    X_test_classical = hstack([X_test_tfidf, X_test_numeric])

    # ---------------------------
    # Initialize scorer
    # ---------------------------
    risk_scorer = RiskScorer()

    # ---------------------------
    # Evaluate models
    # ---------------------------
    prob_lr = evaluate_model(
        "Logistic Regression (TF-IDF)",
        lr_baseline,
        X_test_tfidf,
        y_test,
        risk_scorer
    )

    prob_xgb = evaluate_model(
        "XGBoost (TF-IDF + Manipulation)",
        xgb_model,
        X_test_classical,
        y_test,
        risk_scorer
    )

    # ---------------------------
    # SHAP Explainability (XGBoost)
    # ---------------------------
    print("\nRunning SHAP explainability for XGBoost...")

    tfidf_features = tfidf_vectorizer.get_feature_names_out()
    numeric_features = fb.feature_names

    all_features = list(tfidf_features) + list(numeric_features)

    explainer = XGBExplainer(xgb_model, all_features)

    global_imp = explainer.global_importance(X_test_classical)
    print("\nTop 20 Important Features (Global):")
    print(global_imp.head(20))

    local_exp = explainer.explain_instance(X_test_classical, 0)
    print("\nTop Features for Sample 0:")
    print(local_exp.head(15))

    # ---------------------------
    # BERT Evaluation — raw text
    # ---------------------------
    print("\nRunning BERT inference...")

    prob_bert = evaluate_model(
        "Fine-tuned BERT",
        bert_classifier,
        X_test_raw.tolist(),  
        y_test,
        risk_scorer,
        threshold=0.4
    )

    # ---------------------------
    # ROC Curve
    # ---------------------------
    plt.figure()

    for prob, label in [
        (prob_lr, "TF-IDF LR"),
        (prob_xgb, "XGBoost"),
        (prob_bert, "BERT")
    ]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        plt.plot(fpr, tpr, label=label)

    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    # ---------------------------
    # Precision-Recall Curve
    # ---------------------------
    plt.figure()

    for prob, label in [
        (prob_lr, "TF-IDF LR"),
        (prob_xgb, "XGBoost"),
        (prob_bert, "BERT")        # ← just add this line
    ]:
        p, r, _ = precision_recall_curve(y_test, prob)
        plt.plot(r, p, label=label)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()

    # ---------------------------    
    # Ensemble Voting
    # ---------------------------
    print("\n===== Ensemble (Majority Vote) =====")

    ensemble_prob = (prob_lr + prob_xgb + prob_bert) / 3
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)

    print("Accuracy:",  accuracy_score(y_test, ensemble_pred))
    print("Precision:", precision_score(y_test, ensemble_pred))
    print("Recall:",    recall_score(y_test, ensemble_pred))
    print("F1-score:",  f1_score(y_test, ensemble_pred))
    print("ROC-AUC:",   roc_auc_score(y_test, ensemble_prob))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, ensemble_pred))

    risk_labels = risk_scorer.score_batch(ensemble_prob)
    print("\nRisk Distribution (Ensemble):")
    print(pd.Series(risk_labels).value_counts())


if __name__ == "__main__":
    main()
    

