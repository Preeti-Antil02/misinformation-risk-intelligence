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
from src.models.roberta_model import RobertaClassifier
from src.models.slm_model import QwenClassifier
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
    precision_recall_curve,
)

from scipy.sparse import hstack


# -------------------------------------------------------
# How many samples to run Qwen on (it's slow on CPU)
# Set to None to run on full test set
# -------------------------------------------------------
QWEN_SUBSET = 300


def evaluate_model(name, model, X_test, y_test, risk_scorer, threshold=0.5):

    y_prob = model.predict_proba(X_test)

    if len(y_prob.shape) > 1:
        y_prob = y_prob[:, 1]

    y_pred = (y_prob >= threshold).astype(int)

    risk_labels = risk_scorer.score_batch(y_prob)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    roc  = roc_auc_score(y_test, y_prob)

    print(f"\n===== {name} =====")
    print("Accuracy:",  acc)
    print("Precision:", prec)
    print("Recall:",    rec)
    print("F1-score:",  f1)
    print("ROC-AUC:",   roc)
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

    df["raw_text"] = df["text"].copy()

    tp = TextPreprocessor()
    df["text"] = df["text"].apply(tp.basic_clean)
    df["text"] = df["text"].apply(tp.truncate)

    X = df["text"]
    y = df["label"]
    print("\nLabel distribution:")
    print(y.value_counts())

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_test_raw = df.loc[X_test_text.index, "raw_text"]

    # ---------------------------
    # Load saved models
    # ---------------------------
    lr_baseline = joblib.load("models/baseline_logistic.pkl")
    xgb_model   = joblib.load("models/xgboost_model.pkl")

    print("\nLoading fine-tuned RoBERTa model...")
    roberta_classifier = RobertaClassifier()
    roberta_classifier.load("models/roberta_finetuned")

    tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    scaler           = joblib.load("models/numeric_scaler.pkl")

    # ---------------------------
    # TF-IDF Features
    # ---------------------------
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

    # ---------------------------
    # Manipulation Features
    # ---------------------------
    fb = FeatureBuilder()

    test_df_raw          = df.loc[X_test_text.index].copy()
    test_df_raw["text"]  = test_df_raw["raw_text"]
    X_test_numeric       = fb.build_features(test_df_raw)
    X_test_numeric       = scaler.transform(X_test_numeric)

    X_test_classical = hstack([X_test_tfidf, X_test_numeric])

    # ---------------------------
    # Initialize scorer
    # ---------------------------
    risk_scorer = RiskScorer()

    # ---------------------------
    # Evaluate classical models
    # ---------------------------
    prob_lr = evaluate_model(
        "Logistic Regression (TF-IDF)",
        lr_baseline,
        X_test_tfidf,
        y_test,
        risk_scorer,
    )

    prob_xgb = evaluate_model(
        "XGBoost (TF-IDF + Manipulation)",
        xgb_model,
        X_test_classical,
        y_test,
        risk_scorer,
    )

    # ---------------------------
    # SHAP Explainability (XGBoost)
    # ---------------------------
    print("\nRunning SHAP explainability for XGBoost...")

    tfidf_features   = tfidf_vectorizer.get_feature_names_out()
    numeric_features = fb.feature_names
    all_features     = list(tfidf_features) + list(numeric_features)

    explainer  = XGBExplainer(xgb_model, all_features)
    global_imp = explainer.global_importance(X_test_classical)
    print("\nTop 20 Important Features (Global):")
    print(global_imp.head(20))

    local_exp = explainer.explain_instance(X_test_classical, 0)
    print("\nTop Features for Sample 0:")
    print(local_exp.head(15))

    # ---------------------------
    # Evaluate RoBERTa
    # ---------------------------
    print("\nRunning RoBERTa inference...")
    prob_roberta = evaluate_model(
        "Fine-tuned RoBERTa",
        roberta_classifier,
        X_test_raw.tolist(),
        y_test,
        risk_scorer,
        threshold=0.4,
    )

    # ---------------------------
    # Evaluate Qwen (subset)
    # ---------------------------
    print(f"\nLoading Qwen2.5-3B-Instruct (zero-shot)...")
    qwen_clf = QwenClassifier()

    test_raw_list = X_test_raw.tolist()
    y_test_arr    = np.array(y_test)

    if QWEN_SUBSET and QWEN_SUBSET < len(test_raw_list):
        print(f"[INFO] Running Qwen on {QWEN_SUBSET} samples (subset) to save time.")
        subset_idx   = np.random.RandomState(42).choice(
            len(test_raw_list), QWEN_SUBSET, replace=False
        )
        qwen_texts   = [test_raw_list[i] for i in subset_idx]
        y_qwen       = y_test_arr[subset_idx]
        prob_roberta_sub = prob_roberta[subset_idx]
    else:
        qwen_texts       = test_raw_list
        y_qwen           = y_test_arr
        prob_roberta_sub = prob_roberta

    print("\nRunning Qwen zero-shot inference...")
    prob_qwen = qwen_clf.predict_proba(qwen_texts)
    pred_qwen = (prob_qwen >= 0.5).astype(int)
    risk_qwen = risk_scorer.score_batch(prob_qwen)

    print("\n===== Qwen2.5-3B-Instruct (Zero-Shot) =====")
    print("Accuracy:",  accuracy_score(y_qwen, pred_qwen))
    print("Precision:", precision_score(y_qwen, pred_qwen))
    print("Recall:",    recall_score(y_qwen, pred_qwen))
    print("F1-score:",  f1_score(y_qwen, pred_qwen))
    print("ROC-AUC:",   roc_auc_score(y_qwen, prob_qwen))
    print("Confusion Matrix:")
    print(confusion_matrix(y_qwen, pred_qwen))

    print("\nSample Risk Scores (Qwen):")
    for i in range(min(10, len(prob_qwen))):
        print(f"Prob: {prob_qwen[i]:.4f} → Risk: {risk_qwen[i]}")

    print("\nRisk Distribution (Qwen):")
    print(pd.Series(risk_qwen).value_counts())

    # ---------------------------
    # RoBERTa vs Qwen comparison
    # ---------------------------
    print("\n" + "=" * 60)
    print("RoBERTa vs Qwen2.5 — HEAD-TO-HEAD COMPARISON")
    print("=" * 60)

    roberta_pred_sub = (prob_roberta_sub >= 0.4).astype(int)

    comparison = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
        "RoBERTa (fine-tuned)": [
            round(accuracy_score(y_qwen, roberta_pred_sub),    4),
            round(precision_score(y_qwen, roberta_pred_sub),   4),
            round(recall_score(y_qwen, roberta_pred_sub),      4),
            round(f1_score(y_qwen, roberta_pred_sub),          4),
            round(roc_auc_score(y_qwen, prob_roberta_sub),     4),
        ],
        "Qwen2.5-3B (zero-shot)": [
            round(accuracy_score(y_qwen, pred_qwen),    4),
            round(precision_score(y_qwen, pred_qwen),   4),
            round(recall_score(y_qwen, pred_qwen),      4),
            round(f1_score(y_qwen, pred_qwen),          4),
            round(roc_auc_score(y_qwen, prob_qwen),     4),
        ],
    })

    print(comparison.to_string(index=False))
    if QWEN_SUBSET:
        print(f"\n[Note] Qwen evaluated on {QWEN_SUBSET}-sample subset. RoBERTa shown on same subset.")

    agreement = (roberta_pred_sub == pred_qwen).mean()
    print(f"\nPrediction agreement rate: {agreement:.1%}")

    roberta_conf = np.mean(np.abs(prob_roberta_sub - 0.5)) * 2
    qwen_conf    = np.mean(np.abs(prob_qwen - 0.5)) * 2
    print(f"Average RoBERTa confidence: {roberta_conf:.4f}")
    print(f"Average Qwen confidence:    {qwen_conf:.4f}")

    disagree_mask = roberta_pred_sub != pred_qwen
    print(f"\nDisagreement count: {disagree_mask.sum()} / {len(y_qwen)}")
    if disagree_mask.sum() > 0:
        print(f"RoBERTa correct on disagreements: "
              f"{accuracy_score(y_qwen[disagree_mask], roberta_pred_sub[disagree_mask]):.1%}")
        print(f"Qwen correct on disagreements:    "
              f"{accuracy_score(y_qwen[disagree_mask], pred_qwen[disagree_mask]):.1%}")

    roberta_risk_dist = risk_scorer.score_batch(prob_roberta_sub)
    print("\nRoBERTa Risk Distribution:")
    print(pd.Series(roberta_risk_dist).value_counts())
    print("\nQwen Risk Distribution:")
    print(pd.Series(risk_qwen).value_counts())

    # ---------------------------
    # ROC Curve — all models
    # ---------------------------
    plt.figure(figsize=(8, 6))
    for prob, label in [
        (prob_lr,      "TF-IDF LR"),
        (prob_xgb,     "XGBoost"),
        (prob_roberta, "RoBERTa"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        plt.plot(fpr, tpr, label=label)

    # Qwen on subset
    fpr_q, tpr_q, _ = roc_curve(y_qwen, prob_qwen)
    plt.plot(fpr_q, tpr_q, label=f"Qwen2.5-3B (n={len(y_qwen)})", linestyle="--")

    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — All Models")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # Precision-Recall Curve
    # ---------------------------
    plt.figure(figsize=(8, 6))
    for prob, label in [
        (prob_lr,      "TF-IDF LR"),
        (prob_xgb,     "XGBoost"),
        (prob_roberta, "RoBERTa"),
    ]:
        p, r, _ = precision_recall_curve(y_test, prob)
        plt.plot(r, p, label=label)

    p_q, r_q, _ = precision_recall_curve(y_qwen, prob_qwen)
    plt.plot(r_q, p_q, label=f"Qwen2.5-3B (n={len(y_qwen)})", linestyle="--")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — All Models")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # RoBERTa vs Qwen bar chart
    # ---------------------------
    metrics        = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    roberta_scores = comparison["RoBERTa (fine-tuned)"].tolist()
    qwen_scores    = comparison["Qwen2.5-3B (zero-shot)"].tolist()

    x     = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, roberta_scores, width, label="RoBERTa (fine-tuned)", color="#58a6ff")
    ax.bar(x + width / 2, qwen_scores,    width, label="Qwen2.5-3B (zero-shot)", color="#e3b341")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("RoBERTa vs Qwen2.5-3B — Performance Comparison")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # Ensemble (LR + XGBoost + RoBERTa) — on full test set
    # ---------------------------
    print("\n===== Ensemble (LR + XGBoost + RoBERTa) =====")

    ensemble_prob = (prob_lr * 0.20) + (prob_xgb * 0.40) + (prob_roberta * 0.40)
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