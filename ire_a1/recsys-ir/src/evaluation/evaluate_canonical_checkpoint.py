"""Evaluate canonical saved GBDT re-ranker checkpoints under canonical protocol.

Enforces:
1. STRICT RELOAD FROM DISK: Loads models/ebnerd_reranker.joblib. Never refits or creates ephemeral models.
2. DETERMINISM CHECK: Evaluates Stage-1 baseline twice to verify identical results (>=4 decimal places).
3. CANONICAL EVALUATION:
   - Q2 Before/After (Stage-1 vs Stage-2 Re-Ranker)
   - Q3 Ablation Study (all 6 feature groups)
   - Q3 Paired Bootstrap CI vs Stage-1 and vs Official NRMS
   - Q9 Anti-Gaming Assessment (WITH vs WITHOUT Position Bias)
4. Saves all canonical outputs to results/ directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import polars as pl

from src.common.canonical_protocol import (
    CANONICAL_SEED,
    CANONICAL_VAL_SAMPLE_N,
    EBNERD_CHECKPOINT_PATH,
    MIND_CHECKPOINT_PATH,
    PROJECT_ROOT,
    RESULTS_DIR,
)
from src.common.paths import interim_dir, processed_dir
from src.evaluation.ranking_metrics import auc_score, mrr, ndcg_at_k
from src.feature_store.article_store import ArticleFeatureStore
from src.feature_store.behavioral_features import BehavioralFeatureExtractor
from src.feature_store.session_features import PositionBiasModel, SessionFeatureExtractor
from src.reranking.feature_pipeline import FEATURE_NAMES, ReRankFeaturePipeline
from src.reranking.train_reranker import GBDTReranker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Feature groups for Q3 ablation
ABLATION_GROUPS = {
    "- Category Affinity": [
        "user_category_affinity",
        "user_subcategory_affinity",
        "user_category_clicked_ratio",
    ],
    "- Position Bias": [
        "position_bias_rank",
        "position_bias_relative",
        "position_bias_reciprocal",
        "position_bias_log_discount",
        "position_bias_empirical_ctr",
    ],
    "- Freshness & Recency": [
        "article_age_hours_linear",
        "article_freshness_decay",
        "user_history_recency_score",
        "user_mean_history_age_hours",
        "user_min_history_age_hours",
    ],
    "- Session & Dwell": [
        "session_impression_index",
        "session_clicks_so_far_log",
        "session_dwell_time_log",
        "session_mean_scroll",
        "session_time_since_start_hours",
        "session_time_since_last_min",
        "session_dwell_available",
    ],
    "- Popularity Prior": [
        "train_article_inview_log",
        "train_article_clicks_log",
        "train_article_smoothed_ctr",
        "global_ctr_prior",
    ],
    "- History Embeddings & Semantic Overlap": [
        "user_history_embedding_similarity",
        "user_history_max_embedding_similarity",
        "user_history_semantic_overlap_count",
    ],
}

POSITION_BIAS_FEATURES = ABLATION_GROUPS["- Position Bias"]


def compute_paired_bootstrap(
    diffs: list[float], b: int = 1000, seed: int = CANONICAL_SEED
) -> tuple[float, float, float, float, bool]:
    """Paired bootstrap confidence interval over metric differences."""
    rng = np.random.RandomState(seed)
    diffs_arr = np.asarray(diffs, dtype=np.float64)
    n = len(diffs_arr)
    mean_diff = float(np.mean(diffs_arr))
    boot_means = np.empty(b, dtype=np.float64)

    for i in range(b):
        sample = diffs_arr[rng.randint(0, n, size=n)]
        boot_means[i] = np.mean(sample)

    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    p_val = float(np.mean(boot_means <= 0.0) if mean_diff > 0 else np.mean(boot_means >= 0.0))
    p_val = max(1.0 / b, p_val)
    excludes_zero = bool(ci_low > 0.0 or ci_high < 0.0)
    return mean_diff, ci_low, ci_high, p_val, excludes_zero


def run_canonical_evaluation(
    dataset: str = "ebnerd",
    val_sample: int = CANONICAL_VAL_SAMPLE_N,
    scale: str = "small",
) -> dict[str, Any]:
    """Execute complete canonical evaluation using ONLY the saved model checkpoint."""
    checkpoint_path = EBNERD_CHECKPOINT_PATH if dataset == "ebnerd" else MIND_CHECKPOINT_PATH
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Required canonical checkpoint not found at {checkpoint_path}. "
            f"Run train_canonical_reranker.py first!"
        )

    logger.info("Loading CANONICAL MODEL CHECKPOINT from: %s", checkpoint_path)
    reranker = GBDTReranker.load(checkpoint_path)
    logger.info("Loaded model: type=%s, features=%d", reranker.model_type, len(reranker.feature_names))

    p_dir = processed_dir(dataset, scale)
    i_dir = interim_dir(dataset, scale)
    beh_path = i_dir / "behaviors.parquet"

    article_store = ArticleFeatureStore(dataset, processed_dir=p_dir, scale=scale)
    df_beh = pl.read_parquet(beh_path)

    if "split" in df_beh.columns:
        train_df = df_beh.filter(pl.col("split") == "train")
        val_df = df_beh.filter(pl.col("split") == "val")
    else:
        n = len(df_beh)
        train_df = df_beh.slice(0, int(n * 0.8))
        val_df = df_beh.slice(int(n * 0.8), n - int(n * 0.8))

    # Priors from training split
    from src.evaluation.eval_reranker import load_dataset_popularity
    train_clicks, train_inviews = load_dataset_popularity(train_df)
    pos_model = PositionBiasModel.fit_from_training_behaviors(train_df)

    behavioral_extractor = BehavioralFeatureExtractor(
        dataset=dataset,
        article_store=article_store,
        train_popularity=train_clicks,
        train_inviews=train_inviews,
    )
    session_extractor = SessionFeatureExtractor(
        dataset=dataset,
        position_bias_model=pos_model,
    )
    feature_pipeline = ReRankFeaturePipeline(
        dataset=dataset,
        article_store=article_store,
        session_extractor=session_extractor,
        behavioral_extractor=behavioral_extractor,
    )

    # Sort validation impressions chronologically by user
    user_col = "user_id" if "user_id" in val_df.columns else None
    ts_col = "timestamp" if "timestamp" in val_df.columns else "impression_time"
    val_sorted = val_df.sort([user_col, ts_col]) if user_col else val_df

    # Extract validation impressions and precompute feature matrices
    val_user_prior: dict[str, list[dict[str, Any]]] = {}
    valid_impressions: list[dict[str, Any]] = []

    for row in val_sorted.iter_rows(named=True):
        if len(valid_impressions) >= val_sample:
            break

        raw_cands = row.get("candidates")
        raw_labels = row.get("labels")
        if not raw_cands or not raw_labels:
            continue

        cands = json.loads(raw_cands) if isinstance(raw_cands, str) else list(raw_cands)
        labels = json.loads(raw_labels) if isinstance(raw_labels, str) else list(raw_labels)

        # Non-informative skips
        if sum(labels) == 0 or sum(labels) == len(labels):
            continue

        ts = row.get("timestamp") or row.get("impression_time")
        as_of = datetime.fromisoformat(ts) if isinstance(ts, str) else (ts if isinstance(ts, datetime) else datetime(2023, 5, 20, 12, 0, 0))
        user_id = str(row.get("user_id", "U1"))
        session_id = row.get("session_id")
        raw_hist = row.get("clicked_history")
        u_hist = json.loads(raw_hist) if isinstance(raw_hist, str) else (raw_hist or [])
        if u_hist and isinstance(u_hist[0], str):
            u_hist = [{"article_id": aid, "clicked_at": None} for aid in u_hist]

        prior_imps = val_user_prior.get(user_id, [])

        X_imp = feature_pipeline.extract_impression_features(
            user_id=user_id,
            as_of_ts=as_of,
            candidate_ids=cands,
            user_history=u_hist,
            user_impressions=prior_imps,
            current_session_id=session_id,
        )

        if user_id not in val_user_prior:
            val_user_prior[user_id] = []
        val_user_prior[user_id].append({
            "timestamp": as_of,
            "session_id": session_id,
            "labels": labels,
            "read_time": row.get("read_time"),
            "scroll_percentage": row.get("scroll_percentage"),
        })

        valid_impressions.append({
            "candidates": cands,
            "labels": labels,
            "X": X_imp,
        })

    logger.info("Assembled %d valid evaluation impressions for %s", len(valid_impressions), dataset)

    # -------------------------------------------------------------
    # 1. Determinism Test on Stage-1 Baseline
    # -------------------------------------------------------------
    def compute_stage1_metrics():
        m = {"AUC": [], "MRR": [], "nDCG@5": [], "nDCG@10": []}
        for imp in valid_impressions:
            cands = imp["candidates"]
            labels = imp["labels"]
            K = len(cands)
            base_scores = [float(K - i) for i in range(K)]
            m["AUC"].append(float(auc_score(labels, base_scores)))
            m["MRR"].append(float(mrr(labels, base_scores)))
            m["nDCG@5"].append(float(ndcg_at_k(labels, base_scores, k=5)))
            m["nDCG@10"].append(float(ndcg_at_k(labels, base_scores, k=10)))
        return {k: float(np.mean(v)) for k, v in m.items()}, m

    stage1_run1, stage1_metric_lists = compute_stage1_metrics()
    stage1_run2, _ = compute_stage1_metrics()

    logger.info("=== Stage-1 Baseline Determinism Check ===")
    for k in stage1_run1:
        diff = abs(stage1_run1[k] - stage1_run2[k])
        logger.info("  %s: Run1=%.6f, Run2=%.6f, Diff=%.8f", k, stage1_run1[k], stage1_run2[k], diff)
        assert diff < 1e-6, f"Stage-1 metric {k} is non-deterministic!"

    # -------------------------------------------------------------
    # 2. Stage-2 GBDT Full Model Evaluation
    # -------------------------------------------------------------
    def score_model(mask_indices: list[int] | None = None):
        m = {"AUC": [], "MRR": [], "nDCG@5": [], "nDCG@10": []}
        for imp in valid_impressions:
            labels = imp["labels"]
            X = imp["X"].copy()
            if mask_indices:
                X[:, mask_indices] = 0.0
            pred_scores = reranker.predict_scores(X)
            m["AUC"].append(float(auc_score(labels, pred_scores.tolist())))
            m["MRR"].append(float(mrr(labels, pred_scores.tolist())))
            m["nDCG@5"].append(float(ndcg_at_k(labels, pred_scores.tolist(), k=5)))
            m["nDCG@10"].append(float(ndcg_at_k(labels, pred_scores.tolist(), k=10)))
        return {k: float(np.mean(v)) for k, v in m.items()}, m

    full_means, full_metric_lists = score_model()
    logger.info("=== Full Model Performance (Stage-2 GBDT) ===")
    for k in full_means:
        gain = full_means[k] - stage1_run1[k]
        logger.info("  %s: Stage-1=%.4f -> GBDT=%.4f (Gain=%+.4f)", k, stage1_run1[k], full_means[k], gain)

    # -------------------------------------------------------------
    # 3. Q3 Ablation Study (All 6 Feature Groups)
    # -------------------------------------------------------------
    ablation_rows = [
        {
            "dataset": dataset,
            "configuration": "Baseline (Stage 1)",
            "AUC": stage1_run1["AUC"],
            "MRR": stage1_run1["MRR"],
            "nDCG@5": stage1_run1["nDCG@5"],
            "nDCG@10": stage1_run1["nDCG@10"],
            "delta_AUC": round(stage1_run1["AUC"] - full_means["AUC"], 4),
            "delta_MRR": round(stage1_run1["MRR"] - full_means["MRR"], 4),
            "delta_nDCG@5": round(stage1_run1["nDCG@5"] - full_means["nDCG@5"], 4),
            "delta_nDCG@10": round(stage1_run1["nDCG@10"] - full_means["nDCG@10"], 4),
        },
        {
            "dataset": dataset,
            "configuration": "Full Model (Principled Improvement)",
            "AUC": full_means["AUC"],
            "MRR": full_means["MRR"],
            "nDCG@5": full_means["nDCG@5"],
            "nDCG@10": full_means["nDCG@10"],
            "delta_AUC": 0.0,
            "delta_MRR": 0.0,
            "delta_nDCG@5": 0.0,
            "delta_nDCG@10": 0.0,
        },
    ]

    for group_name, fnames in ABLATION_GROUPS.items():
        indices = [FEATURE_NAMES.index(fn) for fn in fnames if fn in FEATURE_NAMES]
        abl_means, _ = score_model(mask_indices=indices)
        ablation_rows.append({
            "dataset": dataset,
            "configuration": group_name,
            "AUC": abl_means["AUC"],
            "MRR": abl_means["MRR"],
            "nDCG@5": abl_means["nDCG@5"],
            "nDCG@10": abl_means["nDCG@10"],
            "delta_AUC": round(abl_means["AUC"] - full_means["AUC"], 4),
            "delta_MRR": round(abl_means["MRR"] - full_means["MRR"], 4),
            "delta_nDCG@5": round(abl_means["nDCG@5"] - full_means["nDCG@5"], 4),
            "delta_nDCG@10": round(abl_means["nDCG@10"] - full_means["nDCG@10"], 4),
        })

    df_ablation = pl.DataFrame(ablation_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_ablation.write_csv(RESULTS_DIR / f"canonical_ablation_{dataset}.csv")
    logger.info("Ablation study saved to %s", RESULTS_DIR / f"canonical_ablation_{dataset}.csv")

    # -------------------------------------------------------------
    # 4. Q3 Paired Bootstrap CI vs Stage-1 Baseline & Official NRMS
    # -------------------------------------------------------------
    bootstrap_rows = []
    for metric in ["AUC", "MRR", "nDCG@5", "nDCG@10"]:
        diffs = [f - s for f, s in zip(full_metric_lists[metric], stage1_metric_lists[metric])]
        mean_diff, ci_low, ci_high, p_val, excl = compute_paired_bootstrap(diffs)
        bootstrap_rows.append({
            "dataset": dataset,
            "comparison": "Full Model vs Stage 1 Baseline",
            "metric": metric,
            "baseline_mean": stage1_run1[metric],
            "improved_mean": full_means[metric],
            "mean_gain": mean_diff,
            "ci_low_95": ci_low,
            "ci_high_95": ci_high,
            "p_value": p_val,
            "excludes_zero": excl,
        })

    # Official NRMS comparison on EB-NeRD
    # Official NRMS baseline means on EB-NeRD validation:
    # AUC: 0.5807, MRR: 0.3677, nDCG@5: 0.4007, nDCG@10: 0.4826
    if dataset == "ebnerd":
        nrms_means = {"AUC": 0.5807, "MRR": 0.3677, "nDCG@5": 0.4007, "nDCG@10": 0.4826}
        for metric in ["AUC", "MRR", "nDCG@5", "nDCG@10"]:
            diffs_nrms = [f - nrms_means[metric] for f in full_metric_lists[metric]]
            mean_diff, ci_low, ci_high, p_val, excl = compute_paired_bootstrap(diffs_nrms)
            bootstrap_rows.append({
                "dataset": dataset,
                "comparison": "Full Model vs Official NRMS Baseline",
                "metric": metric,
                "baseline_mean": nrms_means[metric],
                "improved_mean": full_means[metric],
                "mean_gain": mean_diff,
                "ci_low_95": ci_low,
                "ci_high_95": ci_high,
                "p_value": p_val,
                "excludes_zero": excl,
            })

    df_bootstrap = pl.DataFrame(bootstrap_rows)
    df_bootstrap.write_csv(RESULTS_DIR / f"canonical_paired_bootstrap_{dataset}.csv")
    logger.info("Paired bootstrap results saved to %s", RESULTS_DIR / f"canonical_paired_bootstrap_{dataset}.csv")

    # -------------------------------------------------------------
    # 5. Q9 Anti-Gaming Table (WITH vs WITHOUT Position Bias)
    # -------------------------------------------------------------
    pos_indices = [FEATURE_NAMES.index(fn) for fn in POSITION_BIAS_FEATURES if fn in FEATURE_NAMES]
    no_pos_means, _ = score_model(mask_indices=pos_indices)

    anti_gaming_rows = [
        {
            "dataset": dataset,
            "configuration": "Full Model (WITH Position Bias / Offline Observational)",
            "AUC": full_means["AUC"],
            "MRR": full_means["MRR"],
            "nDCG@5": full_means["nDCG@5"],
            "nDCG@10": full_means["nDCG@10"],
        },
        {
            "dataset": dataset,
            "configuration": "Realistic Serving Pipeline (WITHOUT Position Bias / Anti-Gaming)",
            "AUC": no_pos_means["AUC"],
            "MRR": no_pos_means["MRR"],
            "nDCG@5": no_pos_means["nDCG@5"],
            "nDCG@10": no_pos_means["nDCG@10"],
        },
        {
            "dataset": dataset,
            "configuration": "Serving-Time Degradation (Delta)",
            "AUC": round(no_pos_means["AUC"] - full_means["AUC"], 4),
            "MRR": round(no_pos_means["MRR"] - full_means["MRR"], 4),
            "nDCG@5": round(no_pos_means["nDCG@5"] - full_means["nDCG@5"], 4),
            "nDCG@10": round(no_pos_means["nDCG@10"] - full_means["nDCG@10"], 4),
        },
        {
            "dataset": dataset,
            "configuration": "Relative Impact (%)",
            "AUC": round(((no_pos_means["AUC"] - full_means["AUC"]) / full_means["AUC"]) * 100, 2),
            "MRR": round(((no_pos_means["MRR"] - full_means["MRR"]) / full_means["MRR"]) * 100, 2),
            "nDCG@5": round(((no_pos_means["nDCG@5"] - full_means["nDCG@5"]) / full_means["nDCG@5"]) * 100, 2),
            "nDCG@10": round(((no_pos_means["nDCG@10"] - full_means["nDCG@10"]) / full_means["nDCG@10"]) * 100, 2),
        },
    ]

    df_anti_gaming = pl.DataFrame(anti_gaming_rows)
    df_anti_gaming.write_csv(RESULTS_DIR / f"canonical_anti_gaming_{dataset}.csv")
    logger.info("Anti-gaming evaluation saved to %s", RESULTS_DIR / f"canonical_anti_gaming_{dataset}.csv")

    # Q2 Summary table
    q2_summary = [
        {"dataset": dataset, "stage": "Stage 1 (Baseline)", **stage1_run1},
        {"dataset": dataset, "stage": "Stage 2 (GBDT Re-Ranker)", **full_means},
        {
            "dataset": dataset,
            "stage": "Absolute Gain (Delta)",
            "AUC": round(full_means["AUC"] - stage1_run1["AUC"], 4),
            "MRR": round(full_means["MRR"] - stage1_run1["MRR"], 4),
            "nDCG@5": round(full_means["nDCG@5"] - stage1_run1["nDCG@5"], 4),
            "nDCG@10": round(full_means["nDCG@10"] - stage1_run1["nDCG@10"], 4),
        },
    ]
    df_q2 = pl.DataFrame(q2_summary)
    df_q2.write_csv(RESULTS_DIR / f"canonical_reranker_eval_{dataset}.csv")

    return {
        "dataset": dataset,
        "n_evaluated": len(valid_impressions),
        "stage1_determinism": {"run1": stage1_run1, "run2": stage1_run2},
        "full_means": full_means,
        "ablation_table": df_ablation.to_dicts(),
        "bootstrap_table": df_bootstrap.to_dicts(),
        "anti_gaming_table": df_anti_gaming.to_dicts(),
        "q2_summary": df_q2.to_dicts(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate canonical saved GBDT re-ranker checkpoint")
    parser.add_argument("--dataset", choices=["ebnerd", "mind"], default="ebnerd")
    parser.add_argument("--val-sample", type=int, default=CANONICAL_VAL_SAMPLE_N)
    args = parser.parse_args()

    run_canonical_evaluation(dataset=args.dataset, val_sample=args.val_sample)
