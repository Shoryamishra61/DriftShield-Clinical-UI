"""
DriftShield Statistical Drift Testing.

This module implements statistical significance tests for concept drift,
specifically the two-sample Kolmogorov-Smirnov (KS) test and the
Population Stability Index (PSI) for monitoring score distribution shifts.
"""

import numpy as np
from scipy.stats import ks_2samp
from typing import Dict, Any, List, Tuple

def calculate_ks_test(reference_scores: List[float], target_scores: List[float]) -> Tuple[float, float, bool]:
    """Performs a two-sample Kolmogorov-Smirnov test on reference and target score distributions.

    Args:
        reference_scores: List of scores representing baseline model behavior.
        target_scores: List of scores representing current monitored query behavior.

    Returns:
        A tuple of:
          - ks_statistic (float): The maximum vertical distance between cumulative distributions.
          - p_value (float): Two-tailed p-value.
          - is_drifted (bool): True if the p-value is below significance threshold (0.05).
    """
    ref = np.array(reference_scores, dtype=np.float32)
    tgt = np.array(target_scores, dtype=np.float32)
    
    if len(ref) == 0 or len(tgt) == 0:
        return 0.0, 1.0, False
        
    stat, p_val = ks_2samp(ref, tgt)
    is_drifted = bool(p_val < 0.05)
    return float(stat), float(p_val), is_drifted


def calculate_psi(reference_scores: List[float], target_scores: List[float], num_bins: int = 10) -> Tuple[float, str]:
    """Calculates the Population Stability Index (PSI) between two score distributions.

    Args:
        reference_scores: List of scores from the reference/baseline distribution.
        target_scores: List of scores from the target/monitored distribution.
        num_bins: Number of bins to bucket the scores into.

    Returns:
        A tuple containing:
          - psi_value (float): The calculated Population Stability Index.
          - interpretation (str): Explanation of the PSI value ('stable', 'moderate shift', 'significant drift').
    """
    ref = np.array(reference_scores, dtype=np.float32)
    tgt = np.array(target_scores, dtype=np.float32)
    
    if len(ref) == 0 or len(tgt) == 0:
        return 0.0, "insufficient_data"

    # Define bins based on reference quantiles to ensure equal count per bin in baseline
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(ref, percentiles)
    # Adjust edges slightly to avoid floating point boundary issues
    bins[0] -= 1e-5
    bins[-1] += 1e-5
    
    # Calculate counts in each bin
    ref_counts, _ = np.histogram(ref, bins=bins)
    tgt_counts, _ = np.histogram(tgt, bins=bins)
    
    # Convert to fractions with Laplace smoothing to avoid division by zero
    ref_probs = (ref_counts + 1e-5) / (len(ref) + 1e-5 * num_bins)
    tgt_probs = (tgt_counts + 1e-5) / (len(tgt) + 1e-5 * num_bins)
    
    # PSI calculation formula
    psi_value = float(np.sum((tgt_probs - ref_probs) * np.log(tgt_probs / ref_probs)))
    
    if psi_value < 0.10:
        interpretation = "stable"
    elif psi_value < 0.25:
        interpretation = "moderate shift"
    else:
        interpretation = "significant drift"
        
    return psi_value, interpretation


def monitor_population_drift(
    reference_scores: List[float],
    target_scores: List[float],
    ks_alpha: float = 0.05,
    psi_threshold: float = 0.25
) -> Dict[str, Any]:
    """Analyzes reference and target query populations for concept drift.

    Args:
        reference_scores: Baseline period scores.
        target_scores: Current monitored period scores.
        ks_alpha: Significance level for the KS test (default 0.05).
        psi_threshold: Threshold for flagging PSI drift (default 0.25).

    Returns:
        A dictionary containing detailed statistics and final alert triggers.
    """
    stat, p_val, ks_drift = calculate_ks_test(reference_scores, target_scores)
    psi_val, psi_interpret = calculate_psi(reference_scores, target_scores)
    
    psi_drift = bool(psi_val >= psi_threshold)
    trigger_retrain = ks_drift or psi_drift

    return {
        "ks_statistic": stat,
        "ks_p_value": p_val,
        "ks_drift_detected": ks_drift,
        "psi_value": psi_val,
        "psi_interpretation": psi_interpret,
        "psi_drift_detected": psi_drift,
        "trigger_retraining": trigger_retrain,
    }
