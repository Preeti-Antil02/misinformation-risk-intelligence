# src/domain_testing.py
# run: python -m src.domain_testing

import os
import joblib
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
)

from src.data_loader import DataLoader
from src.preprocessing import DataCleaner
from src.features.text_preprocessor import TextPreprocessor
from src.features.feature_builder import FeatureBuilder
from src.models.slm_model import QwenClassifier
from src.risk_scoring import RiskScorer

from scipy.sparse import hstack


def run_transformer_inference(model, tokenizer, texts, device,
                               batch_size=32, max_length=256, desc="Inference"):
    model.eval()
    predictions   = []
    probabilities = []

    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = texts[i:i + batch_size]

        encoding = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = torch.softmax(outputs.logits, dim=1)
            preds   = torch.argmax(probs, dim=1)

        predictions.extend(preds.cpu().numpy())
        probabilities.extend(probs[:, 1].cpu().numpy())

    return np.array(predictions), np.array(probabilities)


def run_cross_domain_evaluation():
    print("\n===== CROSS DATASET EVALUATION =====")

    # -------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------
    loader = DataLoader()

    print("Loading FakeNewsNet dataset...")
    df = loader.load_domain_testing()

    print("\nLabel distribution:")
    print(df["label"].value_counts())
    print("Dataset size:", len(df))

    # -------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------
    cleaner = DataCleaner()
    df = cleaner.clean(df)

    texts  = df["text"].astype(str).tolist()
    labels = df["label"].values

    # -------------------------------------------------------
    # Text processing
    # -------------------------------------------------------
    text_processor  = TextPreprocessor()
    texts_processed = [text_processor.advanced_clean(text) for text in texts]

    # -------------------------------------------------------
    # Load TF-IDF vectorizer + scaler
    # -------------------------------------------------------
    print("\nLoading vectorizer...")
    tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    scaler           = joblib.load("models/numeric_scaler.pkl")

    X_tfidf = tfidf_vectorizer.transform(texts_processed)

    # -------------------------------------------------------
    # Manipulation features
    # -------------------------------------------------------
    feature_builder = FeatureBuilder()
    temp_df         = pd.DataFrame({"text": texts})

    manipulation_features = feature_builder.build_features(temp_df)
    manipulation_features = scaler.transform(manipulation_features)

    # -------------------------------------------------------
    # Combine features
    # -------------------------------------------------------
    X = hstack([X_tfidf, manipulation_features])

    # -------------------------------------------------------
    # Load classical models
    # -------------------------------------------------------
    print("\nLoading trained models...")
    lr_model  = joblib.load("models/baseline_logistic.pkl")
    xgb_model = joblib.load("models/xgboost_model.pkl")

    risk_scorer = RiskScorer()

    # -------------------------------------------------------
    # Logistic Regression
    # -------------------------------------------------------
    print("\n===== Logistic Regression (TF-IDF) =====")

    X_lr     = X_tfidf
    lr_preds = lr_model.predict(X_lr)
    lr_probs = lr_model.predict_proba(X_lr)[:, 1]
    lr_risk  = risk_scorer.score_batch(lr_probs)

    print("Accuracy:",  accuracy_score(labels, lr_preds))
    print("Precision:", precision_score(labels, lr_preds))
    print("Recall:",    recall_score(labels, lr_preds))
    print("F1-score:",  f1_score(labels, lr_preds))
    print("ROC-AUC:",   roc_auc_score(labels, lr_probs))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, lr_preds))

    print("\nSample Risk Scores (LR):")
    for i in range(min(10, len(lr_probs))):
        print(f"Prob: {lr_probs[i]:.4f} -> Risk: {lr_risk[i]}")

    # -------------------------------------------------------
    # XGBoost
    # -------------------------------------------------------
    print("\n===== XGBoost (TF-IDF + Manipulation) =====")

    THRESHOLD = 0.75

    xgb_probs = xgb_model.predict_proba(X)[:, 1]
    xgb_probs = np.clip(xgb_probs, 0.05, 0.95)
    xgb_preds = (xgb_probs >= THRESHOLD).astype(int)

    print(f"\nUsing custom threshold for XGBoost: {THRESHOLD}")

    xgb_risk = risk_scorer.score_batch(xgb_probs)

    print("Accuracy:",  accuracy_score(labels, xgb_preds))
    print("Precision:", precision_score(labels, xgb_preds))
    print("Recall:",    recall_score(labels, xgb_preds))
    print("F1-score:",  f1_score(labels, xgb_preds))
    print("ROC-AUC:",   roc_auc_score(labels, xgb_probs))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, xgb_preds))

    print("\nSample Risk Scores (XGB):")
    for i in range(min(10, len(xgb_probs))):
        print(f"Prob: {xgb_probs[i]:.4f} -> Risk: {xgb_risk[i]}")

    default_preds = (xgb_probs >= 0.5).astype(int)
    print("\n--- Threshold Comparison ---")
    print("Default (0.5) Fake count:", default_preds.sum())
    print("Custom Fake count:",        xgb_preds.sum())

    sorted_probs = np.sort(xgb_probs)
    print("\nProbability distribution (XGB):")
    print(sorted_probs[:10], "...", sorted_probs[-10:])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------
    # RoBERTa evaluation
    # -------------------------------------------------------
    if os.path.exists("models/roberta_finetuned"):
        print("\nLoading fine-tuned RoBERTa model...")
        roberta_model     = RobertaForSequenceClassification.from_pretrained("models/roberta_finetuned")
        roberta_tokenizer = RobertaTokenizer.from_pretrained("models/roberta_finetuned")
        roberta_model.to(device)

        print("\nRunning RoBERTa inference...")
        roberta_preds, roberta_probs = run_transformer_inference(
            roberta_model, roberta_tokenizer, texts[:100], device, desc="RoBERTa inference"
        )
        roberta_risk = risk_scorer.score_batch(roberta_probs)

        print("\n===== Fine-tuned RoBERTa =====")
        print("Accuracy:",  accuracy_score(labels[:100], roberta_preds))
        print("Precision:", precision_score(labels[:100], roberta_preds))
        print("Recall:",    recall_score(labels[:100], roberta_preds))
        print("F1-score:",  f1_score(labels[:100], roberta_preds))
        print("ROC-AUC:",   roc_auc_score(labels[:100], roberta_probs))


if __name__ == "__main__":
    run_cross_domain_evaluation()