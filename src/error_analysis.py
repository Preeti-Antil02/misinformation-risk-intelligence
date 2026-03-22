# src/error_analysis.py
# run: python -m src.error_analysis

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from src.data_loader import DataLoader
from src.preprocessing import DataCleaner
from src.features.text_preprocessor import TextPreprocessor
from src.features.feature_builder import FeatureBuilder
from src.models.bert_model import BertClassifier

from scipy.sparse import hstack


EMOTIONAL_WORDS = [
    "breaking", "shocking", "exclusive", "secret",
    "conspiracy", "exposed", "urgent", "alert", "warning"
]

POLITICAL_WORDS = [
    "trump", "clinton", "obama", "biden", "gop",
    "democrat", "republican", "senate", "congress", "white house"
]


# -------------------------------------------------------
# Data loading
# -------------------------------------------------------

def load_data():
    loader = DataLoader()
    df = loader.load_isot()

    cleaner = DataCleaner()
    df = cleaner.remove_duplicates(df)
    df = cleaner.remove_nulls(df)
    df = cleaner.remove_short_texts(df)

    tp = TextPreprocessor()
    df["text"] = df["text"].apply(tp.basic_clean)

    _, X_test_text, _, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
    )

    test_df = df.loc[X_test_text.index].copy()
    return test_df, y_test


# -------------------------------------------------------
# Feature building
# -------------------------------------------------------

def build_features(test_df):
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    scaler = joblib.load("models/numeric_scaler.pkl")

    X_tfidf = tfidf.transform(test_df["text"])

    fb = FeatureBuilder()
    X_num = fb.build_features(test_df)
    X_num = scaler.transform(X_num)

    X_classical = hstack([X_tfidf, X_num])

    return X_tfidf, X_classical


# -------------------------------------------------------
# Model predictions
# -------------------------------------------------------

def get_predictions(test_df, X_tfidf, X_classical):
    lr = joblib.load("models/baseline_logistic.pkl")
    xgb = joblib.load("models/xgboost_model.pkl")

    bert = BertClassifier()
    bert.load("models/bert_finetuned")

    lr_probs = lr.predict_proba(X_tfidf)[:, 1]
    lr_preds = (lr_probs >= 0.5).astype(int)

    xgb_probs = xgb.predict_proba(X_classical)[:, 1]
    xgb_preds = (xgb_probs >= 0.5).astype(int)

    print("\nRunning BERT predictions...")
    bert_probs = bert.predict_proba(test_df["text"].tolist())
    bert_preds = (bert_probs >= 0.5).astype(int)

    return (lr_preds, lr_probs), (xgb_preds, xgb_probs), (bert_preds, bert_probs)


# -------------------------------------------------------
# Error table builder
# -------------------------------------------------------

def build_error_table(test_df, y_test, preds, probs, model_name):
    df = test_df[["text"]].copy()
    df["true_label"] = y_test.values
    df["predicted_label"] = preds
    df["probability"] = probs
    df["model"] = model_name

    df["false_positive"] = (df["true_label"] == 0) & (df["predicted_label"] == 1)
    df["false_negative"] = (df["true_label"] == 1) & (df["predicted_label"] == 0)

    return df


# -------------------------------------------------------
# Pattern analysis — computed from actual data
# -------------------------------------------------------

def count_pattern(texts, word_list):
    return sum(
        1 for t in texts
        if any(w in t.lower() for w in word_list)
    )


def analyze_pattern_bias(fp_texts, fn_texts, total_fp, total_fn):
    results = {}

    for label, texts, total in [
        ("false_positive", fp_texts, total_fp),
        ("false_negative", fn_texts, total_fn),
    ]:
        if total == 0:
            results[label] = {
                "emotional_count": 0, "emotional_pct": 0.0,
                "political_count": 0, "political_pct": 0.0,
                "total": 0
            }
            continue

        emotional = count_pattern(texts, EMOTIONAL_WORDS)
        political = count_pattern(texts, POLITICAL_WORDS)

        results[label] = {
            "emotional_count": emotional,
            "emotional_pct": 100 * emotional / total,
            "political_count": political,
            "political_pct": 100 * political / total,
            "total": total,
        }

    return results


# -------------------------------------------------------
# Per-model error report
# -------------------------------------------------------

def report_model_errors(error_df, model_name):
    fp = error_df[error_df["false_positive"]]
    fn = error_df[error_df["false_negative"]]

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"  False Positives : {len(fp):>5}  (real news predicted as FAKE)")
    print(f"  False Negatives : {len(fn):>5}  (fake news predicted as REAL)")

    print(f"\n  Top 5 False Positives (highest confidence):")
    for _, row in fp.nlargest(5, "probability").iterrows():
        print(f"    {row['probability']:.3f} | {row['text'][:100]}...")

    print(f"\n  Top 5 False Negatives (closest to threshold):")
    for _, row in fn.nsmallest(5, "probability").iterrows():
        print(f"    {row['probability']:.3f} | {row['text'][:100]}...")

    patterns = analyze_pattern_bias(
        fp["text"].tolist(), fn["text"].tolist(), len(fp), len(fn)
    )

    print(f"\n  Pattern bias in False Positives:")
    fp_pat = patterns["false_positive"]
    print(f"    Emotional words : {fp_pat['emotional_count']}/{len(fp)} "
          f"({fp_pat['emotional_pct']:.1f}%)")
    print(f"    Political terms : {fp_pat['political_count']}/{len(fp)} "
          f"({fp_pat['political_pct']:.1f}%)")

    print(f"\n  Pattern bias in False Negatives:")
    fn_pat = patterns["false_negative"]
    print(f"    Emotional words : {fn_pat['emotional_count']}/{len(fn)} "
          f"({fn_pat['emotional_pct']:.1f}%)")
    print(f"    Political terms : {fn_pat['political_count']}/{len(fn)} "
          f"({fn_pat['political_pct']:.1f}%)")

    return fp, fn, patterns


# -------------------------------------------------------
# Error distribution plots — computed
# -------------------------------------------------------

def plot_error_distribution(fp, fn, model_name):
    os.makedirs("outputs", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{model_name} — Error Distribution", fontsize=13)

    if len(fp) > 0:
        axes[0].hist(fp["probability"], bins=20, color="#E24B4A", edgecolor="white")
    axes[0].set_title("False Positives")
    axes[0].set_xlabel("Predicted probability (fake)")
    axes[0].set_ylabel("Count")

    if len(fn) > 0:
        axes[1].hist(fn["probability"], bins=20, color="#378ADD", edgecolor="white")
    axes[1].set_title("False Negatives")
    axes[1].set_xlabel("Predicted probability (fake)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    path = f"outputs/error_dist_{model_name.replace(' ', '_')}.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"\n  Plot saved → {path}")


# -------------------------------------------------------
# Cross-model disagreement analysis
# -------------------------------------------------------

def analyze_disagreements(test_df, y_test, lr_df, xgb_df, bert_df):
    combined = pd.DataFrame({
        "text": test_df["text"].values,
        "true_label": y_test.values,
        "lr_pred": lr_df["predicted_label"].values,
        "lr_prob": lr_df["probability"].values,
        "xgb_pred": xgb_df["predicted_label"].values,
        "xgb_prob": xgb_df["probability"].values,
        "bert_pred": bert_df["predicted_label"].values,
        "bert_prob": bert_df["probability"].values,
    })

    combined["all_agree"] = (
        (combined["lr_pred"] == combined["xgb_pred"]) &
        (combined["xgb_pred"] == combined["bert_pred"])
    )

    disagreements = combined[~combined["all_agree"]]

    print(f"\n{'='*60}")
    print(f"  CROSS-MODEL DISAGREEMENT ANALYSIS")
    print(f"{'='*60}")
    print(f"  Total test samples   : {len(combined)}")
    print(f"  All 3 models agree   : {combined['all_agree'].sum()}")
    print(f"  Models disagree      : {len(disagreements)}")
    print(f"  ({100*len(disagreements)/len(combined):.1f}% of test set is ambiguous)")

    print(f"\n  Sample disagreements (hardest cases):")
    for _, row in disagreements.head(5).iterrows():
        print(f"    True:{int(row['true_label'])} | "
              f"LR:{int(row['lr_pred'])}({row['lr_prob']:.2f}) "
              f"XGB:{int(row['xgb_pred'])}({row['xgb_prob']:.2f}) "
              f"BERT:{int(row['bert_pred'])}({row['bert_prob']:.2f})")
        print(f"    {row['text'][:90]}...")

    os.makedirs("outputs", exist_ok=True)
    combined.to_csv("outputs/error_analysis_full.csv", index=False)
    print(f"\n  Full table saved → outputs/error_analysis_full.csv")

    return combined, disagreements


# -------------------------------------------------------
# Computed findings — generated from actual numbers
# -------------------------------------------------------

def print_computed_findings(lr_patterns, xgb_patterns, bert_patterns,
                             disagreements, combined):
    print(f"\n{'='*60}")
    print(f"  COMPUTED FINDINGS (based on your actual data)")
    print(f"{'='*60}")

    # Political bias
    lr_fp_pol = lr_patterns["false_positive"]["political_pct"]
    xgb_fp_pol = xgb_patterns["false_positive"]["political_pct"]
    bert_fp_pol = bert_patterns["false_positive"]["political_pct"]
    avg_pol = (lr_fp_pol + xgb_fp_pol + bert_fp_pol) / 3

    print(f"\n  1. POLITICAL KEYWORD BIAS IN FALSE POSITIVES")
    print(f"     LR {lr_fp_pol:.1f}%  |  XGB {xgb_fp_pol:.1f}%  |  BERT {bert_fp_pol:.1f}%  "
          f"(avg {avg_pol:.1f}%)")
    if avg_pol > 30:
        print(f"     Finding: HIGH — model flags real political news as misinformation.")
    elif avg_pol > 15:
        print(f"     Finding: MODERATE — political topics increase false positive risk.")
    else:
        print(f"     Finding: LOW — political keywords are not the primary FP driver.")

    # Emotional language
    lr_fp_emo = lr_patterns["false_positive"]["emotional_pct"]
    xgb_fp_emo = xgb_patterns["false_positive"]["emotional_pct"]
    bert_fp_emo = bert_patterns["false_positive"]["emotional_pct"]
    avg_emo = (lr_fp_emo + xgb_fp_emo + bert_fp_emo) / 3

    print(f"\n  2. EMOTIONAL LANGUAGE BIAS IN FALSE POSITIVES")
    print(f"     LR {lr_fp_emo:.1f}%  |  XGB {xgb_fp_emo:.1f}%  |  BERT {bert_fp_emo:.1f}%  "
          f"(avg {avg_emo:.1f}%)")
    if avg_emo > 40:
        print(f"     Finding: HIGH — urgent real news is frequently mislabelled as fake.")
    elif avg_emo > 20:
        print(f"     Finding: MODERATE — emotional tone inflates fake probability.")
    else:
        print(f"     Finding: LOW — emotional words alone do not drive misclassification.")

    # False negative volume
    lr_fn = lr_patterns["false_negative"]["total"]
    xgb_fn = xgb_patterns["false_negative"]["total"]
    bert_fn = bert_patterns["false_negative"]["total"]
    counts = {"LR": lr_fn, "XGB": xgb_fn, "BERT": bert_fn}
    safest = min(counts, key=counts.get)

    print(f"\n  3. FALSE NEGATIVE VOLUME (undetected fake news — dangerous)")
    print(f"     LR {lr_fn}  |  XGB {xgb_fn}  |  BERT {bert_fn}")
    print(f"     Finding: {safest} misses the fewest fake articles → safest for deployment.")

    # Disagreement rate
    dis_pct = 100 * len(disagreements) / len(combined)
    print(f"\n  4. MODEL DISAGREEMENT RATE: {dis_pct:.1f}% of test set")
    if dis_pct > 15:
        print(f"     Finding: HIGH ambiguity. A human-in-the-loop layer is recommended.")
    elif dis_pct > 5:
        print(f"     Finding: MODERATE ambiguity. Ensemble voting would improve reliability.")
    else:
        print(f"     Finding: LOW ambiguity. Models are broadly consistent on this dataset.")

    print(f"\n{'='*60}\n")


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    print("\n===== ERROR ANALYSIS =====")

    test_df, y_test = load_data()
    X_tfidf, X_classical = build_features(test_df)

    (lr_preds, lr_probs), (xgb_preds, xgb_probs), (bert_preds, bert_probs) = \
        get_predictions(test_df, X_tfidf, X_classical)

    lr_df = build_error_table(test_df, y_test, lr_preds, lr_probs, "LR")
    xgb_df = build_error_table(test_df, y_test, xgb_preds, xgb_probs, "XGB")
    bert_df = build_error_table(test_df, y_test, bert_preds, bert_probs, "BERT")

    fp_lr, fn_lr, lr_patterns = report_model_errors(lr_df, "Logistic Regression")
    plot_error_distribution(fp_lr, fn_lr, "Logistic Regression")

    fp_xgb, fn_xgb, xgb_patterns = report_model_errors(xgb_df, "XGBoost")
    plot_error_distribution(fp_xgb, fn_xgb, "XGBoost")

    fp_bert, fn_bert, bert_patterns = report_model_errors(bert_df, "Fine-tuned BERT")
    plot_error_distribution(fp_bert, fn_bert, "BERT")

    combined, disagreements = analyze_disagreements(
        test_df, y_test, lr_df, xgb_df, bert_df
    )

    print_computed_findings(
        lr_patterns, xgb_patterns, bert_patterns, disagreements, combined
    )


if __name__ == "__main__":
    main()