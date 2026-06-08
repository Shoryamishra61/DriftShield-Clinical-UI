"""
DriftShield Evaluation Metrics.

Computes classification and retrieval metrics, conducts McNemar's tests
to check statistical significance, and computes bootstrap confidence intervals.
"""

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix,
)
from scipy.stats import chi2

@dataclass
class ClassificationMetrics:
    """Dataclass holding computed classification quality metrics.

    Attributes:
        accuracy: Accuracy score.
        precision_macro: Macro-averaged precision.
        recall_macro: Macro-averaged recall.
        f1_macro: Macro-averaged F1 score.
        sensitivity: Recall for the positive class (RISKY).
        specificity: Recall for the negative class (SAFE).
        auc_roc: Area Under ROC Curve score.
        mcc: Matthews Correlation Coefficient.
        average_precision: Average Precision score.
        confusion_matrix: A 2D list of shape [[TN, FP], [FN, TP]].
        f1_ci_lower: Lower bound of the 95% confidence interval for Macro F1.
        f1_ci_upper: Upper bound of the 95% confidence interval for Macro F1.
        mcnemar_p_value: P-value from McNemar's significance test.
        mcnemar_significant: True if significance (p < 0.05) is observed.
    """
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    sensitivity: float
    specificity: float
    auc_roc: float
    mcc: float
    average_precision: float
    confusion_matrix: List[List[int]]
    f1_ci_lower: float
    f1_ci_upper: float
    mcnemar_p_value: Optional[float] = None
    mcnemar_significant: Optional[bool] = None


@dataclass
class RetrievalMetrics:
    """Dataclass holding search/retrieval effectiveness metrics.

    Attributes:
        precision_at_5: Average precision of top-5 retrieved chunks.
        bertscore_f1_mean: Mean BERTScore F1 between retrieved and gold guidelines.
        bertscore_f1_std: Standard deviation of the BERTScore F1.
    """
    precision_at_5: float
    bertscore_f1_mean: float
    bertscore_f1_std: float


class EvaluationEngine:
    """Computes, tests, and saves classification and retrieval metrics."""

    def compute_classification_metrics(
        self,
        y_true: List[int],
        y_pred: List[int],
        y_prob: List[float],
        n_bootstrap: int = 1000,
        seed: int = 42,
    ) -> ClassificationMetrics:
        """Computes comprehensive binary classification quality scores with bootstrap CIs.

        Args:
            y_true: List of ground-truth binary labels.
            y_pred: List of predicted binary labels.
            y_prob: List of predicted positive class probabilities.
            n_bootstrap: Number of bootstrap iterations for F1 CI.
            seed: Random seed.

        Returns:
            ClassificationMetrics instance with computed metrics.
        """
        rng = np.random.default_rng(seed)
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        f1_scores: List[float] = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(y_true_arr), size=len(y_true_arr))
            if len(np.unique(y_true_arr[idx])) < 2:
                continue
            f1_scores.append(float(f1_score(y_true_arr[idx], y_pred_arr[idx], average="macro", zero_division=0)))
        f1_scores_arr = np.array(f1_scores)

        # Handle edge case if no successful bootstrap iterations occurred
        if len(f1_scores_arr) == 0:
            f1_ci_lower, f1_ci_upper = 0.0, 0.0
        else:
            f1_ci_lower = float(np.percentile(f1_scores_arr, 2.5))
            f1_ci_upper = float(np.percentile(f1_scores_arr, 97.5))

        # Handle AP calculation
        from sklearn.metrics import average_precision_score
        ap = float(average_precision_score(y_true, y_prob))

        return ClassificationMetrics(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision_macro=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            recall_macro=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            sensitivity=float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
            specificity=float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
            auc_roc=float(roc_auc_score(y_true, y_prob)),
            mcc=float(matthews_corrcoef(y_true, y_pred)),
            average_precision=ap,
            confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
            f1_ci_lower=f1_ci_lower,
            f1_ci_upper=f1_ci_upper,
        )

    def mcnemar_test(
        self,
        y_true: List[int],
        y_pred_a: List[int],
        y_pred_b: List[int],
    ) -> Dict[str, Any]:
        """Runs McNemar's statistical significance test between two models.

        Args:
            y_true: Ground-truth list.
            y_pred_a: Predictions list from baseline model.
            y_pred_b: Predictions list from proposed model.

        Returns:
            Dict containing statistical tests outputs.
        """
        y_true_arr = np.array(y_true)
        a_correct = np.array(y_pred_a) == y_true_arr
        b_correct = np.array(y_pred_b) == y_true_arr
        b01 = int(np.sum(~a_correct & b_correct))
        b10 = int(np.sum(a_correct & ~b_correct))
        n = b01 + b10
        if n == 0:
            return {"statistic": 0.0, "p_value": 1.0, "significant": False, "b01": b01, "b10": b10}
        statistic = float((abs(b01 - b10) - 1) ** 2 / n)
        p_value = float(1 - chi2.cdf(statistic, df=1))
        return {"statistic": statistic, "p_value": p_value, "significant": p_value < 0.05, "b01": b01, "b10": b10}

    def compute_retrieval_metrics(
        self,
        retrieved_chunks: List[List[str]],
        relevant_domains: List[str],
    ) -> RetrievalMetrics:
        """Computes P@5 and other retrieval effectiveness scores.

        Args:
            retrieved_chunks: Nested list of retrieved texts per query.
            relevant_domains: Domain labels for each evaluation sample.

        Returns:
            RetrievalMetrics instance.
        """
        precisions = []
        for chunks, domain in zip(retrieved_chunks, relevant_domains):
            relevant = sum(1 for c in chunks[:5] if domain.lower() in c.lower())
            precisions.append(relevant / min(5, len(chunks)) if chunks else 0.0)
        p_at_5 = float(np.mean(precisions)) if precisions else 0.0
        return RetrievalMetrics(
            precision_at_5=p_at_5,
            bertscore_f1_mean=0.0,
            bertscore_f1_std=0.0,
        )

    def save_metrics(self, metrics: ClassificationMetrics, output_path: Path) -> None:
        """Serializes classification metrics to JSON format.

        Args:
            metrics: Metrics object.
            output_path: Output target file path.
        """
        import dataclasses
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(metrics), f, indent=2)
