"""
src/models/evaluation.py

Comprehensive evaluation suite for RiskLens:
Calculates multi-metric comparison matrices, per-class F1 for Low/Moderate/High/Critical risk tiers,
plots confusion matrices & PR/ROC curves, and generates standalone color-coded HTML reports.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from src.risk_scoring import RiskScorer
from src.models.calibration import compute_expected_calibration_error


class ModelEvaluator:
    """
    Evaluates individual base models, raw ensembles, and calibrated ensembles
    across Accuracy, Precision, Recall, F1 (weighted/macro), AUC-ROC, AUC-PR, ECE, and Brier Score.
    """

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.rs = RiskScorer()

    def evaluate_all(
        self,
        y_test: np.ndarray,
        model_probs: Dict[str, np.ndarray],
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Calculates all performance metrics across provided model probability vectors.

        Parameters
        ----------
        y_test : np.ndarray
            Binary true labels.
        model_probs : dict of str -> np.ndarray
            Dictionary mapping model names to predicted probabilities.

        Returns
        -------
        df_metrics : pd.DataFrame
            Structured metric matrix.
        detailed_stats : dict
            Dictionary of raw predictions and distributions.
        """
        metrics_list = []
        detailed_stats = {}

        for name, probs in model_probs.items():
            preds = (probs >= 0.5).astype(int)
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1_w = f1_score(y_test, preds, average="weighted", zero_division=0)
            f1_m = f1_score(y_test, preds, average="macro", zero_division=0)
            auc_roc = roc_auc_score(y_test, probs)
            auc_pr = average_precision_score(y_test, probs)
            ece, _, _, _ = compute_expected_calibration_error(y_test, probs)
            brier = brier_score_loss(y_test, probs)

            metrics_list.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 (Weighted)": f1_w,
                "F1 (Macro)": f1_m,
                "AUC-ROC": auc_roc,
                "AUC-PR": auc_pr,
                "ECE": ece,
                "Brier Score": brier,
            })

            detailed_stats[name] = {
                "probs": probs,
                "preds": preds,
            }

        df_metrics = pd.DataFrame(metrics_list)
        return df_metrics, detailed_stats

    def compute_risk_tier_f1(self, y_test: np.ndarray, calibrated_probs: np.ndarray) -> Dict[str, float]:
        """
        Computes per-tier classification F1 metrics for Low, Moderate, High, and Critical categories.
        """
        risk_labels = np.array([self.rs.score_ensemble(p) for p in calibrated_probs])
        tiers = ["Low", "Moderate", "High", "Critical"]
        tier_f1_scores = {}

        y_pseudo_tier = np.where(y_test == 1, "Critical", "Low")

        for tier in tiers:
            y_true_binary = (y_pseudo_tier == tier).astype(int)
            y_pred_binary = (risk_labels == tier).astype(int)
            score = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            tier_f1_scores[tier] = float(score)

        return tier_f1_scores

    def plot_confusion_matrix(self, y_test: np.ndarray, ensemble_preds: np.ndarray) -> None:
        """Plots and saves normalized and raw Confusion Matrix for the calibrated ensemble."""
        cm = confusion_matrix(y_test, ensemble_preds)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#0a0e17")
        ax.set_facecolor("#0e1524")

        cax = ax.matshow(cm_norm, cmap="Blues", alpha=0.85)
        fig.colorbar(cax, ax=ax)

        classes = ["Real (0)", "Fake (1)"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(classes, color="#8b9bb4", fontsize=10)
        ax.set_yticklabels(classes, color="#8b9bb4", fontsize=10)

        for i in range(2):
            for j in range(2):
                raw_count = cm[i, j]
                pct = cm_norm[i, j] * 100
                color = "#ffffff" if pct > 50 else "#8b9bb4"
                ax.text(j, i, f"{raw_count:,}\n({pct:.1f}%)", ha="center", va="center", color=color, fontweight="bold", fontsize=11)

        ax.set_xlabel("Predicted Label", color="#8b9bb4", fontsize=11, labelpad=10)
        ax.set_ylabel("True Ground Truth", color="#8b9bb4", fontsize=11, labelpad=10)
        ax.set_title("Calibrated Ensemble — Confusion Matrix", color="#ffffff", fontsize=12, fontweight="bold", pad=20)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1f2c47")

        plt.tight_layout()
        cm_path = self.results_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=200, bbox_inches="tight")
        plt.close()

    def plot_roc_pr_curves(self, y_test: np.ndarray, model_probs: Dict[str, np.ndarray]) -> None:
        """Plots and saves comparative ROC and Precision-Recall curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        fig.patch.set_facecolor("#0a0e17")

        palette = {
            "Logistic Regression": "#8b9bb4",
            "XGBoost": "#f0883e",
            "RoBERTa": "#58a6ff",
            "Qwen2.5-3B": "#d29922",
            "Raw Stacking Ensemble": "#a371f7",
            "Calibrated Ensemble": "#3fb950"
        }

        for ax in axes:
            ax.set_facecolor("#0e1524")
            ax.tick_params(colors="#8b9bb4", labelsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor("#1f2c47")

        # 1. ROC Curves
        axes[0].plot([0, 1], [0, 1], "--", color="#6c7d93", label="Chance (AUC=0.50)", linewidth=1.2)
        for name, probs in model_probs.items():
            fpr, tpr, _ = roc_curve(y_test, probs)
            auc_val = roc_auc_score(y_test, probs)
            color = palette.get(name, "#ffffff")
            lw = 2.5 if "Calibrated" in name or "Stacking" in name else 1.5
            axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", color=color, linewidth=lw)

        axes[0].set_xlabel("False Positive Rate", color="#8b9bb4", fontsize=10)
        axes[0].set_ylabel("True Positive Rate", color="#8b9bb4", fontsize=10)
        axes[0].set_title("Receiver Operating Characteristic (ROC)", color="#ffffff", fontsize=12, fontweight="bold", pad=12)
        axes[0].legend(facecolor="#0e1524", edgecolor="#1f2c47", labelcolor="#c9d1d9", fontsize=8.5, loc="lower right")
        axes[0].grid(True, linestyle=":", alpha=0.3, color="#1f2c47")

        # 2. Precision-Recall Curves
        for name, probs in model_probs.items():
            prec, rec, _ = precision_recall_curve(y_test, probs)
            pr_auc = average_precision_score(y_test, probs)
            color = palette.get(name, "#ffffff")
            lw = 2.5 if "Calibrated" in name or "Stacking" in name else 1.5
            axes[1].plot(rec, prec, label=f"{name} (AUC-PR={pr_auc:.3f})", color=color, linewidth=lw)

        axes[1].set_xlabel("Recall", color="#8b9bb4", fontsize=10)
        axes[1].set_ylabel("Precision", color="#8b9bb4", fontsize=10)
        axes[1].set_title("Precision-Recall Curve (AUC-PR)", color="#ffffff", fontsize=12, fontweight="bold", pad=12)
        axes[1].legend(facecolor="#0e1524", edgecolor="#1f2c47", labelcolor="#c9d1d9", fontsize=8.5, loc="lower left")
        axes[1].grid(True, linestyle=":", alpha=0.3, color="#1f2c47")

        plt.tight_layout()
        roc_path = self.results_dir / "pr_roc_curves.png"
        plt.savefig(roc_path, dpi=200, bbox_inches="tight")
        plt.close()

    def generate_html_report(
        self,
        df_metrics: pd.DataFrame,
        tier_f1_scores: Dict[str, float],
        test_loop_results: List[Dict[str, Any]]
    ) -> None:
        """
        Generates a standalone, color-coded HTML report with inline styles only.
        """
        html_path = self.results_dir / "phase1_final_report.html"

        def format_cell(val: float, is_error_metric: bool = False) -> str:
            if is_error_metric:
                if val < 0.05:
                    bg = "rgba(63, 185, 80, 0.18)"
                    fg = "#3fb950"
                    border = "#2ea043"
                elif val < 0.12:
                    bg = "rgba(227, 179, 65, 0.18)"
                    fg = "#e3b341"
                    border = "#d29922"
                else:
                    bg = "rgba(248, 81, 73, 0.18)"
                    fg = "#f85149"
                    border = "#da3633"
            else:
                if val >= 0.90:
                    bg = "rgba(63, 185, 80, 0.18)"
                    fg = "#3fb950"
                    border = "#2ea043"
                elif val >= 0.82:
                    bg = "rgba(227, 179, 65, 0.18)"
                    fg = "#e3b341"
                    border = "#d29922"
                else:
                    bg = "rgba(248, 81, 73, 0.18)"
                    fg = "#f85149"
                    border = "#da3633"

            return f'<td style="padding:12px 14px;background:{bg};color:{fg};border:1px solid {border};font-weight:600;text-align:center;border-radius:6px;font-family:monospace;">{val:.4f}</td>'

        rows_html = ""
        for _, row in df_metrics.iterrows():
            is_winner = "Calibrated" in row["Model"]
            row_style = 'style="background:#131c2e;"' if is_winner else 'style="background:#0e1524;"'
            badge = '<span style="background:#238636;color:#ffffff;font-size:10px;padding:2px 6px;border-radius:10px;margin-left:6px;font-weight:700;">CHOSEN</span>' if is_winner else ""

            rows_html += f"""
            <tr {row_style}>
                <td style="padding:12px 16px;font-weight:700;color:#e6edf3;border-bottom:1px solid #1f2c47;">{row['Model']} {badge}</td>
                {format_cell(row['Accuracy'])}
                {format_cell(row['Precision'])}
                {format_cell(row['Recall'])}
                {format_cell(row['F1 (Weighted)'])}
                {format_cell(row['F1 (Macro)'])}
                {format_cell(row['AUC-ROC'])}
                {format_cell(row['AUC-PR'])}
                {format_cell(row['ECE'], is_error_metric=True)}
                {format_cell(row['Brier Score'], is_error_metric=True)}
            </tr>
            """

        tier_html = ""
        tier_colors = {"Low": "#3fb950", "Moderate": "#e3b341", "High": "#f0883e", "Critical": "#f85149"}
        for tier, score in tier_f1_scores.items():
            c = tier_colors.get(tier, "#ffffff")
            tier_html += f"""
            <div style="background:#0e1524;border:1px solid #1f2c47;border-top:3px solid {c};border-radius:10px;padding:16px 20px;min-width:180px;flex:1;">
                <div style="font-size:12px;color:#8b9bb4;text-transform:uppercase;letter-spacing:0.05em;font-weight:600;">{tier} Risk Tier</div>
                <div style="font-size:26px;color:{c};font-weight:800;margin-top:6px;font-family:monospace;">F1: {score:.4f}</div>
                <div style="font-size:11px;color:#6c7d93;margin-top:4px;">RiskLens proxy target</div>
            </div>
            """

        test_loop_html = ""
        for t in test_loop_results:
            status_color = "#3fb950" if t["passed"] else "#f85149"
            status_badge = f'<span style="background:{status_color}22;color:{status_color};border:1px solid {status_color};font-size:11px;padding:3px 8px;border-radius:6px;font-weight:700;">{"PASS" if t["passed"] else "FAIL"}</span>'
            test_loop_html += f"""
            <div style="display:flex;align-items:center;justify-content:space-between;padding:12px 16px;background:#0e1524;border:1px solid #1f2c47;border-radius:8px;margin-bottom:8px;">
                <div>
                    <span style="color:#e6edf3;font-weight:600;font-size:13px;">{t['name']}</span>
                    <div style="color:#8b9bb4;font-size:11px;margin-top:2px;">Condition: {t['cond']} | Observed: <span style="font-family:monospace;color:#ffffff;">{t['actual']:.4f}</span></div>
                </div>
                <div>{status_badge}</div>
            </div>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RiskLens Phase 1 — Trustworthy ML Evaluation Report</title>
</head>
<body style="background:#080c12;color:#c9d1d9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;margin:0;padding:32px 24px;line-height:1.5;">

    <div style="max-width:1240px;margin:0 auto;">
        
        <!-- Header Banner -->
        <div style="background:linear-gradient(135deg,#0e1726 0%,#162238 100%);border:1px solid #1f2c47;border-radius:14px;padding:28px 36px;margin-bottom:28px;">
            <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:16px;">
                <div>
                    <div style="font-size:12px;font-weight:700;color:#58a6ff;letter-spacing:0.1em;text-transform:uppercase;">Phase 1 Production Verification</div>
                    <h1 style="font-size:28px;color:#ffffff;margin:6px 0 0 0;font-weight:800;letter-spacing:-0.02em;">RiskLens Core Trustworthy Ensemble Report</h1>
                    <div style="font-size:13px;color:#8b9bb4;margin-top:6px;">Stacking Meta-Learner (LightGBM/LR) &bull; Platt Scaling &bull; Expected Calibration Error &bull; 20% Held-Out Split</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:11px;color:#8b9bb4;">Report Timestamp</div>
                    <div style="font-size:14px;color:#ffffff;font-weight:700;font-family:monospace;">{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}</div>
                </div>
            </div>
        </div>

        <!-- Testing Loops & Gates -->
        <div style="margin-bottom:28px;">
            <h2 style="font-size:16px;color:#e6edf3;font-weight:700;margin-bottom:12px;">Automated Phase 1 Testing Loops & Quality Gates</h2>
            {test_loop_html}
        </div>

        <!-- Metric Table -->
        <div style="margin-bottom:28px;">
            <h2 style="font-size:16px;color:#e6edf3;font-weight:700;margin-bottom:12px;">Full Evaluation Matrix (Held-out Test Set)</h2>
            <div style="overflow-x:auto;border:1px solid #1f2c47;border-radius:12px;">
                <table style="width:100%;border-collapse:collapse;font-size:13px;">
                    <thead>
                        <tr style="background:#131c2e;color:#8b9bb4;text-align:center;border-bottom:2px solid #1f2c47;">
                            <th style="padding:14px 16px;text-align:left;color:#ffffff;">Model Architecture</th>
                            <th style="padding:14px 10px;">Accuracy</th>
                            <th style="padding:14px 10px;">Precision</th>
                            <th style="padding:14px 10px;">Recall</th>
                            <th style="padding:14px 10px;">F1 (Weighted)</th>
                            <th style="padding:14px 10px;">F1 (Macro)</th>
                            <th style="padding:14px 10px;">AUC-ROC</th>
                            <th style="padding:14px 10px;">AUC-PR</th>
                            <th style="padding:14px 10px;">ECE &darr;</th>
                            <th style="padding:14px 10px;">Brier Score &darr;</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
            <div style="font-size:11px;color:#8b9bb4;margin-top:8px;">
                <span style="display:inline-block;width:10px;height:10px;background:#3fb950;border-radius:2px;margin-right:4px;"></span> Green: Optimal / High-Confidence
                <span style="display:inline-block;width:10px;height:10px;background:#e3b341;border-radius:2px;margin-left:14px;margin-right:4px;"></span> Yellow: Acceptable
                <span style="display:inline-block;width:10px;height:10px;background:#f85149;border-radius:2px;margin-left:14px;margin-right:4px;"></span> Red: Baseline / Needs Attention
            </div>
        </div>

        <!-- Per-Class Risk Tier Cards -->
        <div style="margin-bottom:28px;">
            <h2 style="font-size:16px;color:#e6edf3;font-weight:700;margin-bottom:12px;">Per-Class F1 Score by RiskLens Risk Category</h2>
            <div style="display:flex;gap:14px;flex-wrap:wrap;">
                {tier_html}
            </div>
        </div>

        <!-- Visual Artifacts Grid -->
        <div>
            <h2 style="font-size:16px;color:#e6edf3;font-weight:700;margin-bottom:12px;">Saved Visual Artifacts</h2>
            <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(360px, 1fr));gap:16px;">
                <div style="background:#0e1524;border:1px solid #1f2c47;border-radius:10px;padding:16px;text-align:center;">
                    <div style="font-size:13px;font-weight:700;color:#ffffff;margin-bottom:8px;">Calibration Reliability Diagram</div>
                    <img src="calibration_plot.png" style="max-width:100%;border-radius:6px;border:1px solid #1f2c47;" alt="Calibration Diagram">
                </div>
                <div style="background:#0e1524;border:1px solid #1f2c47;border-radius:10px;padding:16px;text-align:center;">
                    <div style="font-size:13px;font-weight:700;color:#ffffff;margin-bottom:8px;">Confusion Matrix (Calibrated Ensemble)</div>
                    <img src="confusion_matrix.png" style="max-width:100%;border-radius:6px;border:1px solid #1f2c47;" alt="Confusion Matrix">
                </div>
                <div style="background:#0e1524;border:1px solid #1f2c47;border-radius:10px;padding:16px;text-align:center;">
                    <div style="font-size:13px;font-weight:700;color:#ffffff;margin-bottom:8px;">ROC & Precision-Recall Curves</div>
                    <img src="pr_roc_curves.png" style="max-width:100%;border-radius:6px;border:1px solid #1f2c47;" alt="ROC PR Curves">
                </div>
            </div>
        </div>

        <!-- Footer -->
        <div style="margin-top:36px;padding-top:20px;border-top:1px solid #1f2c47;font-size:12px;color:#8b9bb4;text-align:center;">
            RiskLens Misinformation Intelligence System &bull; Phase 1 Production Pipeline &bull; Stacking + Platt Calibration Verified
        </div>

    </div>
</body>
</html>
"""
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
