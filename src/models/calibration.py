"""
src/models/calibration.py

Probability calibration engine for RiskLens.
Implements Platt Scaling (Sigmoid), Isotonic Regression, Expected Calibration Error (ECE)
from scratch with 10 equal-width bins, Brier Score, and reliability diagram plotting.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the Expected Calibration Error (ECE) from scratch using equal-width probability bins.

    ECE = sum_{m=1}^{M} ( |B_m| / N ) * | acc(B_m) - conf(B_m) |

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities in [0.0, 1.0].
    n_bins : int, default=10
        Number of equal-width probability bins.

    Returns
    -------
    ece : float
        Scalar Expected Calibration Error value.
    bin_accuracies : np.ndarray
        Empirical accuracy in each bin.
    bin_confidences : np.ndarray
        Mean predicted confidence in each bin.
    bin_counts : np.ndarray
        Number of samples falling into each bin.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])  # maps to 0..n_bins-1

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    total_samples = len(y_true)
    ece = 0.0

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        count = np.sum(mask)
        bin_counts.append(count)

        if count > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            ece += (count / total_samples) * abs(bin_acc - bin_conf)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append((bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2.0)

    return float(ece), np.array(bin_accuracies), np.array(bin_confidences), np.array(bin_counts)


class ProbabilityCalibrator:
    """
    Applies Platt Scaling (Sigmoid) and Isotonic Regression to ensemble outputs,
    evaluates Expected Calibration Error (ECE) and Brier scores, plots reliability diagrams,
    and persists the optimal calibrated model.
    """

    def __init__(self, results_dir: Path, models_dir: Path):
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.calibrated_model: Optional[CalibratedClassifierCV] = None
        self.method_chosen: str = ""

    def calibrate_and_evaluate(
        self,
        meta_learner: Any,
        X_val_meta: np.ndarray,
        y_val: np.ndarray,
        X_test_meta: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Fits and compares Platt Scaling vs Isotonic Regression on calibration validation data.
        Selects the model with the lowest ECE and logs comprehensive diagnostics.
        """
        # Uncalibrated baseline probabilities
        raw_test_probs = meta_learner.predict_proba(X_test_meta)[:, 1]
        raw_ece, _, _, _ = compute_expected_calibration_error(y_test, raw_test_probs)
        raw_brier = brier_score_loss(y_test, raw_test_probs)

        # 1. Platt Scaling (Sigmoid)
        platt_calibrator = CalibratedClassifierCV(estimator=meta_learner, method="sigmoid", cv="prefit")
        platt_calibrator.fit(X_val_meta, y_val)
        platt_test_probs = platt_calibrator.predict_proba(X_test_meta)[:, 1]
        platt_ece, _, _, _ = compute_expected_calibration_error(y_test, platt_test_probs)
        platt_brier = brier_score_loss(y_test, platt_test_probs)

        # 2. Isotonic Regression
        iso_calibrator = CalibratedClassifierCV(estimator=meta_learner, method="isotonic", cv="prefit")
        iso_calibrator.fit(X_val_meta, y_val)
        iso_test_probs = iso_calibrator.predict_proba(X_test_meta)[:, 1]
        iso_ece, _, _, _ = compute_expected_calibration_error(y_test, iso_test_probs)
        iso_brier = brier_score_loss(y_test, iso_test_probs)

        if platt_ece <= iso_ece:
            self.method_chosen = "Platt Scaling (Sigmoid)"
            self.calibrated_model = platt_calibrator
            best_cal_probs = platt_test_probs
            best_ece = platt_ece
            best_brier = platt_brier
        else:
            self.method_chosen = "Isotonic Regression"
            self.calibrated_model = iso_calibrator
            best_cal_probs = iso_test_probs
            best_ece = iso_ece
            best_brier = iso_brier

        # Save calibrated ensemble
        cal_path_models = self.models_dir / "calibrated_ensemble.pkl"
        cal_path_results = self.results_dir / "calibrated_ensemble.pkl"
        joblib.dump(self.calibrated_model, cal_path_models)
        joblib.dump(self.calibrated_model, cal_path_results)

        # Plot reliability diagram
        self.plot_reliability_diagram(
            y_test,
            raw_test_probs,
            platt_test_probs,
            iso_test_probs,
            raw_ece,
            platt_ece,
            iso_ece
        )

        # Log calibration report
        cal_report_path = self.results_dir / "phase1_calibration_report.txt"
        with open(cal_report_path, "w", encoding="utf-8") as f:
            f.write("RiskLens — Phase 1: Probability Calibration Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Selected Calibration Method : {self.method_chosen}\n\n")
            f.write("CALIBRATION METRICS COMPARISON (10 Equal-Width Bins):\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Raw Ensemble         : ECE = {raw_ece:.4f} | Brier Score = {raw_brier:.4f}\n")
            f.write(f"  Platt Scaling        : ECE = {platt_ece:.4f} | Brier Score = {platt_brier:.4f}\n")
            f.write(f"  Isotonic Regression  : ECE = {iso_ece:.4f} | Brier Score = {iso_brier:.4f}\n")
            f.write("-" * 60 + "\n")
            f.write(f"ECE Improvement   : {((raw_ece - best_ece) / (raw_ece + 1e-9)) * 100:.2f}%\n")
            f.write(f"Brier Improvement : {((raw_brier - best_brier) / (raw_brier + 1e-9)) * 100:.2f}%\n\n")
            f.write("Sample Calibrated Continuous Probabilities Output:\n")
            for i in range(min(10, len(best_cal_probs))):
                f.write(f"  Sample {i+1:02d}: Raw = {raw_test_probs[i]:.4f} -> Calibrated = {best_cal_probs[i]:.4f} (True: {int(y_test[i])})\n")

        return {
            "method_chosen": self.method_chosen,
            "raw_ece": raw_ece,
            "raw_brier": raw_brier,
            "best_ece": best_ece,
            "best_brier": best_brier,
            "calibrated_probs": best_cal_probs,
            "raw_probs": raw_test_probs,
            "calibrated_model": self.calibrated_model,
        }

    def plot_reliability_diagram(
        self,
        y_test: np.ndarray,
        raw_probs: np.ndarray,
        platt_probs: np.ndarray,
        iso_probs: np.ndarray,
        raw_ece: float,
        platt_ece: float,
        iso_ece: float
    ) -> None:
        """Plots and saves multi-model calibration curve & probability density distributions."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        fig.patch.set_facecolor("#0a0e17")

        for ax in axes:
            ax.set_facecolor("#0e1524")
            ax.tick_params(colors="#8b9bb4", labelsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor("#1f2c47")

        # 1. Calibration Reliability Curve
        axes[0].plot([0, 1], [0, 1], "--", color="#6c7d93", label="Perfect Calibration (y=x)", linewidth=1.5)

        prob_true_raw, prob_pred_raw = calibration_curve(y_test, raw_probs, n_bins=10)
        axes[0].plot(prob_pred_raw, prob_true_raw, "s-", color="#f85149", label=f"Raw Ensemble (ECE={raw_ece:.3f})", linewidth=2)

        prob_true_platt, prob_pred_platt = calibration_curve(y_test, platt_probs, n_bins=10)
        axes[0].plot(prob_pred_platt, prob_true_platt, "o-", color="#3fb950", label=f"Platt Scaling (ECE={platt_ece:.3f})", linewidth=2)

        prob_true_iso, prob_pred_iso = calibration_curve(y_test, iso_probs, n_bins=10)
        axes[0].plot(prob_pred_iso, prob_true_iso, "^-", color="#58a6ff", label=f"Isotonic (ECE={iso_ece:.3f})", linewidth=2)

        axes[0].set_xlabel("Mean Predicted Probability", color="#8b9bb4", fontsize=10)
        axes[0].set_ylabel("Empirical True Fraction (Accuracy)", color="#8b9bb4", fontsize=10)
        axes[0].set_title("Reliability Diagram (10 Equal Bins)", color="#ffffff", fontsize=12, fontweight="bold", pad=12)
        axes[0].legend(facecolor="#0e1524", edgecolor="#1f2c47", labelcolor="#c9d1d9", fontsize=9, loc="upper left")
        axes[0].grid(True, linestyle=":", alpha=0.3, color="#1f2c47")

        # 2. Probability Distribution Histogram
        axes[1].hist(raw_probs, bins=25, alpha=0.4, color="#f85149", label="Raw Ensemble", edgecolor="none")
        axes[1].hist(platt_probs, bins=25, alpha=0.6, color="#3fb950", label="Platt Calibrated", edgecolor="none")
        axes[1].set_xlabel("Calibrated Probability Score (0.0 – 1.0)", color="#8b9bb4", fontsize=10)
        axes[1].set_ylabel("Count", color="#8b9bb4", fontsize=10)
        axes[1].set_title("Prediction Probability Density", color="#ffffff", fontsize=12, fontweight="bold", pad=12)
        axes[1].legend(facecolor="#0e1524", edgecolor="#1f2c47", labelcolor="#c9d1d9", fontsize=9)
        axes[1].grid(True, linestyle=":", alpha=0.3, color="#1f2c47")

        plt.tight_layout()
        plot_path = self.results_dir / "calibration_plot.png"
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
