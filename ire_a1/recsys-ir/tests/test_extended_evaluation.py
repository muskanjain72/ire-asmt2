"""Unit tests for Question 5 Extended Evaluation harness."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.evaluation.beyond_accuracy import (
    compute_coverage,
    compute_intra_list_diversity,
    compute_novelty,
)
from src.evaluation.extended_evaluation import (
    ExtendedMockArticleStore,
    evaluate_extended_pipeline,
)
from src.evaluation.slicing import get_article_slice, get_user_slice


def test_ild_computation_bounds():
    """Verify ILD outputs valid distance range [0, 2] and handles edge cases."""
    # Single item or empty
    assert compute_intra_list_diversity(np.empty((0, 384))) == 0.0
    assert compute_intra_list_diversity(np.ones((1, 384))) == 0.0

    # Identical embeddings -> diversity 0.0
    vec = np.ones(384)
    identical = np.vstack([vec, vec, vec])
    assert pytest.approx(compute_intra_list_diversity(identical), abs=1e-5) == 0.0

    # Orthogonal embeddings -> cosine sim 0.0 -> diversity 1.0
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    e3 = np.array([0.0, 0.0, 1.0])
    ortho = np.vstack([e1, e2, e3])
    assert pytest.approx(compute_intra_list_diversity(ortho), abs=1e-5) == 1.0


def test_novelty_monotonicity():
    """Verify novelty is strictly higher for rare items than popular items."""
    train_pop = {"popular": 10000, "medium": 500, "rare": 2}
    total_n = 50000

    nov_popular = compute_novelty(["popular"], train_pop, total_n)
    nov_medium = compute_novelty(["medium"], train_pop, total_n)
    nov_rare = compute_novelty(["rare"], train_pop, total_n)

    assert nov_rare > nov_medium > nov_popular
    assert nov_popular > 0.0


def test_coverage_calculation():
    """Verify catalog coverage fraction arithmetic."""
    all_recs = {"A1", "A2", "A3", "A4"}
    catalog_size = 100
    cov = compute_coverage(all_recs, catalog_size)
    assert pytest.approx(cov, abs=1e-5) == 0.04
    assert compute_coverage(set(), 100) == 0.0


def test_slicing_segmentation():
    """Verify user cold/warm and item head/tail segmentation."""
    assert get_user_slice(0, "mind", "fixed") == "cold"
    assert get_user_slice(5, "mind", "fixed") == "cold"
    assert get_user_slice(6, "mind", "fixed") == "warm"

    assert get_article_slice(5.0, "mind", "fixed") == "tail"
    assert get_article_slice(10.0, "mind", "fixed") == "tail"
    assert get_article_slice(10.1, "mind", "fixed") == "head"


def test_extended_evaluation_pipeline_e2e():
    """Verify evaluate_extended_pipeline produces all expected metrics, slices, and CIs."""
    df_metrics, df_slices = evaluate_extended_pipeline(
        dataset="mind",
        scale="small",
        sample_size=100,
        b_bootstrap=50,  # Fast for unit test
    )

    # All metrics check
    assert len(df_metrics) == 7  # AUC, MRR, nDCG@5, nDCG@10, ILD, Novelty, Coverage
    metric_names = df_metrics["metric"].to_list()
    for expected in ["AUC", "MRR", "nDCG@5", "nDCG@10", "ILD", "Novelty", "Coverage"]:
        assert expected in metric_names

    # Check CI bounds
    for row in df_metrics.iter_rows(named=True):
        assert row["ci_low_95"] <= row["mean"] <= row["ci_high_95"]
        assert row["ci_width"] >= 0.0

    # Slices check: 4 slices * 6 metrics = 24 rows
    assert len(df_slices) == 24
    slice_names = set(df_slices["slice_name"].to_list())
    assert "Cold-Start Users (history <= 5)" in slice_names
    assert "Warm Users (history > 5)" in slice_names
    assert "Head Articles (mean pop > 10)" in slice_names
    assert "Tail Articles (mean pop <= 10)" in slice_names
