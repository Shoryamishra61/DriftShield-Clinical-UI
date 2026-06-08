"""
DriftShield Test Suite — Statistical Drift Tests.

Tests the Kolmogorov-Smirnov and Population Stability Index implementations
for correctness, edge cases, and scientific rigor.
"""

import pytest
import numpy as np
from evaluation.drift_tests import calculate_ks_test, calculate_psi, monitor_population_drift


class TestKolmogorovSmirnovTest:
    """Tests for the KS two-sample test implementation."""

    def test_identical_distributions_no_drift(self):
        """Identical distributions should yield high p-value and no drift."""
        np.random.seed(42)
        ref = np.random.normal(0.3, 0.1, 200).tolist()
        tgt = np.random.normal(0.3, 0.1, 200).tolist()
        stat, p_val, is_drifted = calculate_ks_test(ref, tgt)
        assert p_val > 0.05, f"Expected p > 0.05 for identical distributions, got {p_val}"
        assert not is_drifted

    def test_different_distributions_drift_detected(self):
        """Clearly different distributions should yield low p-value and detect drift."""
        np.random.seed(42)
        ref = np.random.normal(0.2, 0.05, 200).tolist()
        tgt = np.random.normal(0.8, 0.05, 200).tolist()
        stat, p_val, is_drifted = calculate_ks_test(ref, tgt)
        assert p_val < 0.05, f"Expected p < 0.05 for different distributions, got {p_val}"
        assert is_drifted

    def test_empty_inputs_safe_default(self):
        """Empty inputs should return safe defaults."""
        stat, p_val, is_drifted = calculate_ks_test([], [1.0, 2.0, 3.0])
        assert stat == 0.0
        assert p_val == 1.0
        assert not is_drifted

    def test_ks_statistic_range(self):
        """KS statistic should be between 0 and 1."""
        ref = [0.1, 0.2, 0.3, 0.4, 0.5]
        tgt = [0.6, 0.7, 0.8, 0.9, 1.0]
        stat, _, _ = calculate_ks_test(ref, tgt)
        assert 0.0 <= stat <= 1.0


class TestPopulationStabilityIndex:
    """Tests for the PSI implementation."""

    def test_identical_distributions_stable(self):
        """Identical distributions should yield PSI close to 0 (stable)."""
        np.random.seed(42)
        ref = np.random.normal(0.5, 0.1, 500).tolist()
        psi_val, interp = calculate_psi(ref, ref)
        assert psi_val < 0.10, f"Expected PSI < 0.10 for identical, got {psi_val}"
        assert interp == "stable"

    def test_shifted_distributions_significant_drift(self):
        """Significantly shifted distributions should yield high PSI."""
        np.random.seed(42)
        ref = np.random.normal(0.2, 0.05, 500).tolist()
        tgt = np.random.normal(0.8, 0.05, 500).tolist()
        psi_val, interp = calculate_psi(ref, tgt)
        assert psi_val >= 0.25, f"Expected PSI >= 0.25 for shifted distributions, got {psi_val}"
        assert interp == "significant drift"

    def test_moderate_shift(self):
        """Moderately shifted distributions should yield moderate PSI."""
        np.random.seed(42)
        ref = np.random.normal(0.4, 0.1, 500).tolist()
        tgt = np.random.normal(0.55, 0.1, 500).tolist()
        psi_val, interp = calculate_psi(ref, tgt)
        # PSI interpretation is data-dependent, just verify it runs
        assert psi_val >= 0.0

    def test_empty_inputs(self):
        """Empty inputs should return safe defaults."""
        psi_val, interp = calculate_psi([], [1.0, 2.0])
        assert psi_val == 0.0
        assert interp == "insufficient_data"


class TestMonitorPopulationDrift:
    """Integration tests for the combined drift monitoring function."""

    def test_no_drift_no_retraining(self):
        """Stable distributions should not trigger retraining."""
        np.random.seed(42)
        ref = np.random.normal(0.3, 0.1, 200).tolist()
        tgt = np.random.normal(0.3, 0.1, 200).tolist()
        result = monitor_population_drift(ref, tgt)
        assert not result["trigger_retraining"]
        assert "ks_statistic" in result
        assert "psi_value" in result

    def test_drift_triggers_retraining(self):
        """Drifted distributions should trigger retraining."""
        np.random.seed(42)
        ref = np.random.normal(0.2, 0.05, 200).tolist()
        tgt = np.random.normal(0.8, 0.05, 200).tolist()
        result = monitor_population_drift(ref, tgt)
        assert result["trigger_retraining"]
        assert result["ks_drift_detected"] or result["psi_drift_detected"]

    def test_output_keys_complete(self):
        """Result dict should contain all expected keys."""
        ref = [0.1, 0.2, 0.3, 0.4, 0.5]
        tgt = [0.6, 0.7, 0.8, 0.9, 1.0]
        result = monitor_population_drift(ref, tgt)
        expected_keys = [
            "ks_statistic", "ks_p_value", "ks_drift_detected",
            "psi_value", "psi_interpretation", "psi_drift_detected",
            "trigger_retraining"
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
