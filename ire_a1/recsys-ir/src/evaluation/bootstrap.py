"""Bootstrap confidence interval estimation for evaluation metrics."""

from __future__ import annotations

import numpy as np

def compute_bootstrap_ci(
    metric_values: np.ndarray,
    b: int = 1000,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.
    
    Vectorized computation over pre-calculated per-impression metrics.
    
    Parameters
    ----------
    metric_values : np.ndarray
        1D array of metric values (one per impression).
    b : int
        Number of bootstrap iterations.
    random_state : int
        Random seed for reproducibility.
        
    Returns
    -------
    tuple[float, float, float]
        (mean, ci_low, ci_high) representing the sample mean and the
        [2.5th, 97.5th] percentiles of the bootstrap distribution.
    """
    if len(metric_values) == 0:
        return 0.0, 0.0, 0.0
        
    rng = np.random.default_rng(random_state)
    n = len(metric_values)
    
    # Generate B bootstrap samples of size N (indices with replacement)
    # Shape: (B, N)
    indices = rng.integers(0, n, size=(b, n))
    
    # Vectorized indexing: grab metric values for all bootstrap samples
    # Shape: (B, N)
    samples = metric_values[indices]
    
    # Compute mean for each bootstrap sample
    # Shape: (B,)
    sample_means = np.mean(samples, axis=1)
    
    # Calculate 2.5th and 97.5th percentiles
    mean_val = float(np.mean(metric_values))
    ci_low = float(np.percentile(sample_means, 2.5))
    ci_high = float(np.percentile(sample_means, 97.5))
    return mean_val, ci_low, ci_high



def compute_paired_bootstrap_ci(
    metric_base: np.ndarray | Sequence[float],
    metric_improved: np.ndarray | Sequence[float],
    b: int = 1000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compute paired bootstrap confidence interval for the metric difference (improved - base).

    Resamples the exact same impression pairs across B iterations to test whether
    the performance gain strictly excludes zero at the 95% confidence level:
        Δ_i = M_improved(i) - M_base(i)
        Δ^(b) = mean(Δ[indices_b])

    Parameters
    ----------
    metric_base : np.ndarray or Sequence[float]
        1D array of per-impression metric values for baseline model.
    metric_improved : np.ndarray or Sequence[float]
        1D array of per-impression metric values for improved model (paired 1-to-1).
    b : int
        Number of bootstrap iterations (default 1000).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        - "mean_base": float
        - "mean_improved": float
        - "mean_diff": float
        - "ci_low": float (2.5th percentile of Δ^(b))
        - "ci_high": float (97.5th percentile of Δ^(b))
        - "p_value": float (empirical two-tailed p-value against null Δ=0)
        - "excludes_zero": bool (True if 0 is outside [ci_low, ci_high])
        - "statistically_significant": bool (True if ci_low > 0.0)
    """
    base_arr = np.asarray(metric_base, dtype=np.float64)
    imp_arr = np.asarray(metric_improved, dtype=np.float64)

    if len(base_arr) != len(imp_arr):
        raise ValueError(
            f"Mismatched array lengths: metric_base has {len(base_arr)} items, "
            f"metric_improved has {len(imp_arr)} items."
        )

    n = len(base_arr)
    if n == 0:
        return {
            "mean_base": 0.0,
            "mean_improved": 0.0,
            "mean_diff": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_value": 1.0,
            "excludes_zero": False,
            "statistically_significant": False,
        }

    diffs = imp_arr - base_arr
    mean_diff = float(np.mean(diffs))
    mean_base = float(np.mean(base_arr))
    mean_imp = float(np.mean(imp_arr))

    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, n, size=(b, n))
    bootstrap_samples = diffs[indices]
    bootstrap_diff_means = np.mean(bootstrap_samples, axis=1)

    ci_low = float(np.percentile(bootstrap_diff_means, 2.5))
    ci_high = float(np.percentile(bootstrap_diff_means, 97.5))

    # Empirical two-tailed p-value
    if mean_diff >= 0:
        p_val = float(2.0 * np.mean(bootstrap_diff_means <= 0.0))
    else:
        p_val = float(2.0 * np.mean(bootstrap_diff_means >= 0.0))
    p_val = min(1.0, max(1.0 / b, p_val))

    excludes_zero = (ci_low > 0.0) or (ci_high < 0.0)
    is_sig = (ci_low > 0.0)

    return {
        "mean_base": mean_base,
        "mean_improved": mean_imp,
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_val,
        "excludes_zero": excludes_zero,
        "statistically_significant": is_sig,
    }

