"""
DriftShield Visualization and Reporting Module.

Implements plot generation for confusion matrices, ROC curves, drift score distributions,
and model performance comparison charts. Also generates final markdown reports.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc

class ReportGenerator:
    """Generates visualization charts and markdown reports for the evaluation run."""

    def __init__(self, output_dir: Path) -> None:
        """Initializes the ReportGenerator.

        Args:
            output_dir: Directory where the generated figures will be saved.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Apply standard clean styling
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.titlesize": 14
        })

    def plot_confusion_matrix(self, cm: List[List[int]], title: str) -> None:
        """Plots and saves the confusion matrix heatmap.

        Args:
            cm: Confusion matrix of shape [[TN, FP], [FN, TP]].
            title: Title of the model configuration.
        """
        plt.figure(figsize=(6, 5))
        cm_arr = np.array(cm)
        # Reformat label positions
        labels = [["TN\n" + str(cm[0][0]), "FP\n" + str(cm[0][1])],
                  ["FN\n" + str(cm[1][0]), "TP\n" + str(cm[1][1])]]
        
        sns.heatmap(
            cm_arr,
            annot=labels,
            fmt="",
            cmap="Blues",
            cbar=False,
            xticklabels=["Predicted SAFE", "Predicted RISKY"],
            yticklabels=["Actual SAFE", "Actual RISKY"]
        )
        plt.title(f"{title} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.png", dpi=300)
        plt.close()

    def plot_roc_curve(self, y_true: List[int], y_prob: List[float]) -> None:
        """Plots and saves the Receiver Operating Characteristic (ROC) curve.

        Args:
            y_true: Ground truth binary labels.
            y_prob: Predicted positive class probabilities.
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(self.output_dir / "roc_curve.png", dpi=300)
        plt.close()

    def plot_drift_score_distribution(self, safe_scores: List[float], risky_scores: List[float]) -> None:
        """Plots the density/histogram distribution of drift scores for SAFE vs RISKY.

        Args:
            safe_scores: Predicted drift scores for true SAFE samples.
            risky_scores: Predicted drift scores for true RISKY samples.
        """
        plt.figure(figsize=(7, 5))
        sns.kdeplot(safe_scores, fill=True, color="green", label="SAFE class", alpha=0.5, bw_adjust=0.5)
        sns.kdeplot(risky_scores, fill=True, color="red", label="RISKY class", alpha=0.5, bw_adjust=0.5)
        plt.axvline(x=0.5, color="black", linestyle="--", label="Threshold (0.50)")
        plt.xlabel("Predicted Drift Score")
        plt.ylabel("Density")
        plt.title("Drift Score Distribution by Actual Class")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(self.output_dir / "drift_score_distribution.png", dpi=300)
        plt.close()

    def plot_model_comparison(self, model_metrics: Any, baseline_metrics: Any) -> None:
        """Plots a bar chart comparing DriftShield model to keyword baseline.

        Args:
            model_metrics: Computed metrics for the DriftShield classifier.
            baseline_metrics: Computed metrics for the keyword-matching baseline.
        """
        metrics = ["Accuracy", "F1 (Macro)", "Sensitivity", "Specificity", "AUC-ROC"]
        model_vals = [
            model_metrics.accuracy,
            model_metrics.f1_macro,
            model_metrics.sensitivity,
            model_metrics.specificity,
            model_metrics.auc_roc
        ]
        base_vals = [
            baseline_metrics.accuracy,
            baseline_metrics.f1_macro,
            baseline_metrics.sensitivity,
            baseline_metrics.specificity,
            baseline_metrics.auc_roc
        ]

        x = np.arange(len(metrics))
        width = 0.35

        _, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width/2, base_vals, width, label="Keyword Baseline", color="#9CA3AF")
        ax.bar(x + width/2, model_vals, width, label="DriftShield (Proposed)", color="#3B82F6")

        ax.set_ylabel("Score")
        ax.set_title("Performance Comparison: DriftShield vs Keyword Baseline")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim([0.0, 1.1])
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "model_comparison.png", dpi=300)
        plt.close()

    def generate_markdown_report(
        self,
        model_metrics: Any,
        baseline_metrics: Any,
        mcnemar_result: Dict[str, Any],
        results_dir: Path
    ) -> None:
        """Generates a markdown report summarizing the evaluation findings.

        Args:
            model_metrics: Computed proposed model metrics.
            baseline_metrics: Computed baseline model metrics.
            mcnemar_result: McNemar significance test results dict.
            results_dir: Root results directory.
        """
        report_path = Path(results_dir) / "evaluation_report.md"
        report_content = f"""# DriftShield Model Evaluation Report

This report summarizes the comparative evaluation run between the proposed **DriftShield Classifier** (BioBERT fine-tuned sequence classifier with FAISS guideline context) and the **Keyword Matching Baseline**.

## Summary of Results

| Metric | proposed (DriftShield) | Keyword Baseline | Improvement |
|---|---|---|---|
| **Accuracy** | {model_metrics.accuracy:.2%} | {baseline_metrics.accuracy:.2%} | {model_metrics.accuracy - baseline_metrics.accuracy:+.2%} |
| **F1 (Macro)** | {model_metrics.f1_macro:.4f} | {baseline_metrics.f1_macro:.4f} | {model_metrics.f1_macro - baseline_metrics.f1_macro:+.4f} |
| **Sensitivity** | {model_metrics.sensitivity:.2%} | {baseline_metrics.sensitivity:.2%} | {model_metrics.sensitivity - baseline_metrics.sensitivity:+.2%} |
| **Specificity** | {model_metrics.specificity:.2%} | {baseline_metrics.specificity:.2%} | {model_metrics.specificity - baseline_metrics.specificity:+.2%} |
| **AUC-ROC** | {model_metrics.auc_roc:.4f} | {baseline_metrics.auc_roc:.4f} | {model_metrics.auc_roc - baseline_metrics.auc_roc:+.4f} |
| **MCC** | {model_metrics.mcc:.4f} | {baseline_metrics.mcc:.4f} | {model_metrics.mcc - baseline_metrics.mcc:+.4f} |

### Statistical Significance (McNemar's Test)
- **Statistic value**: `{mcnemar_result['statistic']:.4f}`
- **p-value**: `{mcnemar_result['p_value']:.4f}`
- **Statistically Significant**: `{"Yes" if mcnemar_result['significant'] else "No"}` (p < 0.05 threshold)

### Macro F1 95% Confidence Interval
- **proposed Model F1 Range**: `[{model_metrics.f1_ci_lower:.4f}, {model_metrics.f1_ci_upper:.4f}]` (bootstrapped over 1,000 iterations)

## Visualizations

All evaluation plots are generated and saved under `results/figures/`:
1. **Confusion Matrix**: `confusion_matrix.png`
2. **ROC Curve**: `roc_curve.png`
3. **Drift Score Density**: `drift_score_distribution.png`
4. **Model Comparison**: `model_comparison.png`
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"Markdown evaluation report saved to {report_path}")
