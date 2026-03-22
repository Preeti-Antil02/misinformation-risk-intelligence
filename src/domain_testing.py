# src/domain_testing.py

import joblib
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from transformers import BertTokenizer, BertForSequenceClassification

from src.data_loader import DataLoader
from src.preprocessing import DataCleaner
from src.features.text_preprocessor import TextPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.features.feature_builder import FeatureBuilder
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

texts = df["text"].astype(str).tolist()
labels = df["label"].values


# -------------------------------------------------------
# Text processing
# -------------------------------------------------------

text_processor = TextPreprocessor()
texts_processed = [text_processor.advanced_clean(text) for text in texts]


# -------------------------------------------------------
# Load trained TF-IDF vectorizer
# -------------------------------------------------------

print("\nLoading vectorizer...")

tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
scaler = joblib.load("models/numeric_scaler.pkl") 

X_tfidf = tfidf_vectorizer.transform(texts_processed)


# -------------------------------------------------------
# Manipulation features (FeatureBuilder)
# -------------------------------------------------------

feature_builder = FeatureBuilder()

# FeatureBuilder expects a dataframe with a "text" column
temp_df = pd.DataFrame({"text": texts})

manipulation_features = feature_builder.build_features(temp_df)
manipulation_features = scaler.transform(manipulation_features)

# -------------------------------------------------------
# Combine features
# -------------------------------------------------------

X = hstack([X_tfidf, manipulation_features])


# -------------------------------------------------------
# Load trained models
# -------------------------------------------------------

print("\nLoading trained models...")

lr_model = joblib.load("models/baseline_logistic.pkl")
xgb_model = joblib.load("models/xgboost_model.pkl")

risk_scorer = RiskScorer()


print("\n===== Logistic Regression (TF-IDF) =====")

X_lr = X_tfidf

lr_preds = lr_model.predict(X_lr)
lr_probs = lr_model.predict_proba(X_lr)[:, 1]

lr_risk = risk_scorer.score_batch(lr_probs)

print("Accuracy:", accuracy_score(labels, lr_preds))
print("Precision:", precision_score(labels, lr_preds))
print("Recall:", recall_score(labels, lr_preds))
print("F1-score:", f1_score(labels, lr_preds))
print("ROC-AUC:", roc_auc_score(labels, lr_probs))
print("Confusion Matrix:")
print(confusion_matrix(labels, lr_preds))

print("\nSample Risk Scores (LR):")
for i in range(10):
    print(f"Prob: {lr_probs[i]:.4f} → Risk: {lr_risk[i]}")

# -------------------------------------------------------
# XGBoost
# -------------------------------------------------------

print("\n===== XGBoost (TF-IDF + Manipulation) =====")

THRESHOLD = 0.75   # you can tune between 0.7–0.85

xgb_probs = xgb_model.predict_proba(X)[:, 1]
# Fix overconfidence (calibration hack)
xgb_probs = np.clip(xgb_probs, 0.05, 0.95)
xgb_preds = (xgb_probs >= THRESHOLD).astype(int)

print(f"\nUsing custom threshold for XGBoost: {THRESHOLD}")

xgb_risk = risk_scorer.score_batch(xgb_probs)

print("Accuracy:", accuracy_score(labels, xgb_preds))
print("Precision:", precision_score(labels, xgb_preds))
print("Recall:", recall_score(labels, xgb_preds))
print("F1-score:", f1_score(labels, xgb_preds))
print("ROC-AUC:", roc_auc_score(labels, xgb_probs))
print("Confusion Matrix:")
print(confusion_matrix(labels, xgb_preds))

print("\nSample Risk Scores (XGB):")
for i in range(10):
    print(f"Prob: {xgb_probs[i]:.4f} → Risk: {xgb_risk[i]}")

# default predictions (for comparison)
default_preds = (xgb_probs >= 0.5).astype(int)

print("\n--- Threshold Comparison ---")

print("Default (0.5) Fake count:", default_preds.sum())
print("Custom Fake count:", xgb_preds.sum())

# Analyze probability distribution:
sorted_probs = np.sort(xgb_probs)

print("\nProbability distribution (XGB):")
print(sorted_probs[:10], "...", sorted_probs[-10:])

# -------------------------------------------------------
# BERT evaluation
# -------------------------------------------------------

print("\nLoading fine-tuned BERT model...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_model = BertForSequenceClassification.from_pretrained("models/bert_finetuned")
tokenizer = BertTokenizer.from_pretrained("models/bert_finetuned")

bert_model.to(device)
bert_model.eval()


# -------------------------------------------------------
# BERT inference
# -------------------------------------------------------

print("\nRunning BERT inference...")

batch_size = 32

predictions = []
probabilities = []

for i in tqdm(range(0, len(texts), batch_size), desc="BERT inference"):

    batch_texts = texts[i:i+batch_size]

    encoding = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=256,   # increased for better context
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():

        outputs = bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

        preds = torch.argmax(probs, dim=1)

        predictions.extend(preds.cpu().numpy())
        probabilities.extend(probs[:,1].cpu().numpy())

bert_probs = np.array(probabilities)
bert_risk = risk_scorer.score_batch(bert_probs)


# -------------------------------------------------------
# BERT metrics
# -------------------------------------------------------

print("\n===== Fine-tuned BERT =====")

print("Accuracy:", accuracy_score(labels, predictions))
print("Precision:", precision_score(labels, predictions))
print("Recall:", recall_score(labels, predictions))
print("F1-score:", f1_score(labels, predictions))
print("ROC-AUC:", roc_auc_score(labels, bert_probs))
print("Confusion Matrix:")
print(confusion_matrix(labels, predictions))

print("\nSample Risk Scores (BERT):")
for i in range(10):
    print(f"Prob: {bert_probs[i]:.4f} → Risk: {bert_risk[i]}")

# -------------------------------------------------------
# Saving Output
# -------------------------------------------------------

print("\nSaving risk-scored results...")

results_df = pd.DataFrame({
    "text": texts,
    "true_label": labels,
    "bert_probability": bert_probs,
    "bert_risk": bert_risk
})

results_df.to_csv("outputs/domain_risk_results.csv", index=False)

print("Saved to outputs/domain_risk_results.csv")