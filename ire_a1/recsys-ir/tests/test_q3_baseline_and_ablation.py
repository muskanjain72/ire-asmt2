"""Unit tests for Question 3 baseline reproduction, paired bootstrap CI, and ablations."""

import numpy as np
import pytest

from src.evaluation.ablation_study import ABLATION_MASKS, apply_feature_mask
from src.evaluation.bootstrap import compute_paired_bootstrap_ci
from src.reranking.feature_pipeline import FEATURE_NAMES


def test_paired_bootstrap_ci_significant_gain():
    """Verify paired bootstrap CI strictly excludes zero for genuine improvements."""
    np.random.seed(42)
    n = 200
    base = np.random.normal(0.5, 0.1, size=n)
    # Improved is strictly better by +0.08 on average
    improved = base + np.random.normal(0.08, 0.03, size=n)

    res = compute_paired_bootstrap_ci(base, improved, b=500, random_state=42)

    assert res["mean_diff"] > 0.05
    assert res["ci_low"] > 0.0  # Strictly excludes zero!
    assert res["ci_high"] > res["ci_low"]
    assert res["excludes_zero"] is True
    assert res["statistically_significant"] is True
    assert res["p_value"] < 0.05


def test_paired_bootstrap_ci_null_difference():
    """Verify paired bootstrap CI includes zero when models are identical."""
    np.random.seed(42)
    n = 150
    base = np.random.normal(0.5, 0.1, size=n)
    identical = base.copy()

    res = compute_paired_bootstrap_ci(base, identical, b=500, random_state=42)

    assert res["mean_diff"] == 0.0
    assert res["ci_low"] <= 0.0 <= res["ci_high"]
    assert res["statistically_significant"] is False


def test_paired_bootstrap_ci_degradation():
    """Verify paired bootstrap CI correctly flags negative differences."""
    np.random.seed(42)
    n = 200
    base = np.random.normal(0.6, 0.1, size=n)
    worse = base - np.random.normal(0.06, 0.02, size=n)

    res = compute_paired_bootstrap_ci(base, worse, b=500, random_state=42)

    assert res["mean_diff"] < -0.04
    assert res["ci_high"] < 0.0  # Both bounds negative
    assert res["excludes_zero"] is True
    assert res["statistically_significant"] is False


def test_apply_feature_mask():
    """Verify feature masking zeros out requested columns without mutating input."""
    X = np.ones((5, len(FEATURE_NAMES)), dtype=np.float32)
    mask = [0, 1, 24]
    X_masked = apply_feature_mask(X, mask)

    assert (X == 1.0).all()  # Original unchanged
    assert (X_masked[:, mask] == 0.0).all()
    # Unmasked columns remain 1.0
    unmasked = [i for i in range(len(FEATURE_NAMES)) if i not in mask]
    assert (X_masked[:, unmasked] == 1.0).all()


def test_ablation_masks_valid_indices():
    """Verify all indices in ABLATION_MASKS match valid feature names."""
    total_features = len(FEATURE_NAMES)
    for name, indices in ABLATION_MASKS.items():
        for idx in indices:
            assert 0 <= idx < total_features, f"Index {idx} in {name} exceeds {total_features}"
