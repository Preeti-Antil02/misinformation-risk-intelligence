# src/domain_testing.py
# run: python -m src.domain_testing

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
for i in range(10):
    print(f"Prob: {lr_probs[i]:.4f} → Risk: {lr_risk[i]}")

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
for i in range(10):
    print(f"Prob: {xgb_probs[i]:.4f} → Risk: {xgb_risk[i]}")

default_preds = (xgb_probs >= 0.5).astype(int)
print("\n--- Threshold Comparison ---")
print("Default (0.5) Fake count:", default_preds.sum())
print("Custom Fake count:",        xgb_preds.sum())

sorted_probs = np.sort(xgb_probs)
print("\nProbability distribution (XGB):")
print(sorted_probs[:10], "...", sorted_probs[-10:])

# -------------------------------------------------------
# Helper — transformer (encoder) inference
# -------------------------------------------------------

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------
# RoBERTa evaluation
# -------------------------------------------------------

print("\nLoading fine-tuned RoBERTa model...")

roberta_model     = RobertaForSequenceClassification.from_pretrained("models/roberta_finetuned")
roberta_tokenizer = RobertaTokenizer.from_pretrained("models/roberta_finetuned")
roberta_model.to(device)

print("\nRunning RoBERTa inference...")
roberta_preds, roberta_probs = run_transformer_inference(
    roberta_model, roberta_tokenizer, texts, device, desc="RoBERTa inference"
)
roberta_risk = risk_scorer.score_batch(roberta_probs)

print("\n===== Fine-tuned RoBERTa =====")
print("Accuracy:",  accuracy_score(labels, roberta_preds))
print("Precision:", precision_score(labels, roberta_preds))
print("Recall:",    recall_score(labels, roberta_preds))
print("F1-score:",  f1_score(labels, roberta_preds))
print("ROC-AUC:",   roc_auc_score(labels, roberta_probs))
print("Confusion Matrix:")
print(confusion_matrix(labels, roberta_preds))

print("\nSample Risk Scores (RoBERTa):")
for i in range(10):
    print(f"Prob: {roberta_probs[i]:.4f} → Risk: {roberta_risk[i]}")

# -------------------------------------------------------
# Qwen2.5-3B zero-shot evaluation
# NOTE: Runs one sample at a time — slow on CPU (~2-3 sec/sample)
# For full 23k samples this will take hours. Use a subset for testing.
# Set EVAL_SUBSET below to limit samples for quick evaluation.
# -------------------------------------------------------

EVAL_SUBSET = 500   # set to None to run on full dataset

if EVAL_SUBSET:
    print(f"\n[INFO] Evaluating Qwen on subset of {EVAL_SUBSET} samples to save time.")
    subset_idx    = np.random.choice(len(texts), EVAL_SUBSET, replace=False)
    qwen_texts    = [texts[i] for i in subset_idx]
    qwen_labels   = labels[subset_idx]
else:
    qwen_texts  = texts
    qwen_labels = labels

print("\nLoading Qwen2.5-3B-Instruct (zero-shot)...")
qwen_clf = QwenClassifier()

print("\nRunning Qwen zero-shot inference...")
qwen_probs = qwen_clf.predict_proba(qwen_texts)
qwen_preds = (qwen_probs >= 0.5).astype(int)
qwen_risk  = risk_scorer.score_batch(qwen_probs)

print("\n===== Qwen2.5-3B-Instruct (Zero-Shot) =====")
print("Accuracy:",  accuracy_score(qwen_labels, qwen_preds))
print("Precision:", precision_score(qwen_labels, qwen_preds))
print("Recall:",    recall_score(qwen_labels, qwen_preds))
print("F1-score:",  f1_score(qwen_labels, qwen_preds))
print("ROC-AUC:",   roc_auc_score(qwen_labels, qwen_probs))
print("Confusion Matrix:")
print(confusion_matrix(qwen_labels, qwen_preds))

print("\nSample Risk Scores (Qwen):")
for i in range(min(10, len(qwen_texts))):
    print(f"Prob: {qwen_probs[i]:.4f} → Risk: {qwen_risk[i]}")

# -------------------------------------------------------
# RoBERTa vs Qwen — cross-domain comparison
# (Qwen evaluated on subset — note in comparison)
# -------------------------------------------------------

print("\n" + "=" * 60)
print("RoBERTa vs Qwen2.5 — CROSS-DOMAIN COMPARISON (FakeNewsNet)")
print("=" * 60)

if EVAL_SUBSET:
    # Compare on same subset
    roberta_probs_sub = roberta_probs[subset_idx]
    roberta_preds_sub = roberta_preds[subset_idx]
else:
    roberta_probs_sub = roberta_probs
    roberta_preds_sub = roberta_preds

comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
    "RoBERTa (fine-tuned)": [
        round(accuracy_score(qwen_labels, roberta_preds_sub),  4),
        round(precision_score(qwen_labels, roberta_preds_sub), 4),
        round(recall_score(qwen_labels, roberta_preds_sub),    4),
        round(f1_score(qwen_labels, roberta_preds_sub),        4),
        round(roc_auc_score(qwen_labels, roberta_probs_sub),   4),
    ],
    "Qwen2.5-3B (zero-shot)": [
        round(accuracy_score(qwen_labels, qwen_preds),    4),
        round(precision_score(qwen_labels, qwen_preds),   4),
        round(recall_score(qwen_labels, qwen_preds),      4),
        round(f1_score(qwen_labels, qwen_preds),          4),
        round(roc_auc_score(qwen_labels, qwen_probs),     4),
    ],
})

print(comparison.to_string(index=False))
if EVAL_SUBSET:
    print(f"\n[Note] Qwen evaluated on {EVAL_SUBSET}-sample subset. RoBERTa shown on same subset.")

agreement = (roberta_preds_sub == qwen_preds).mean()
print(f"\nPrediction agreement rate: {agreement:.1%}")

print("\nRoBERTa Risk Distribution (cross-domain):")
print(pd.Series(roberta_risk).value_counts())
print("\nQwen Risk Distribution (cross-domain):")
print(pd.Series(qwen_risk).value_counts())

# -------------------------------------------------------
# Save output
# -------------------------------------------------------

print("\nSaving risk-scored results...")

# Full RoBERTa results
results_df = pd.DataFrame({
    "text":                  texts,
    "true_label":            labels,
    "roberta_probability":   roberta_probs,
    "roberta_risk":          roberta_risk,
})

# Add Qwen results (subset or full)
if EVAL_SUBSET:
    qwen_prob_col = np.full(len(texts), np.nan)
    qwen_risk_col = ["N/A"] * len(texts)
    for j, idx in enumerate(subset_idx):
        qwen_prob_col[idx] = qwen_probs[j]
        qwen_risk_col[idx] = qwen_risk[j]
    results_df["qwen_probability"] = qwen_prob_col
    results_df["qwen_risk"]        = qwen_risk_col
else:
    results_df["qwen_probability"] = qwen_probs
    results_df["qwen_risk"]        = qwen_risk

results_df.to_csv("outputs/domain_risk_results.csv", index=False)
print("Saved to outputs/domain_risk_results.csv")