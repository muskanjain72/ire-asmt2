"""Train and save canonical GBDT re-ranker checkpoints under the canonical protocol.

Enforces:
1. Deterministic random seed (CANONICAL_SEED = 42).
2. Training on train-split behaviors with chronological user-session grouping (no leakage).
3. Non-zero session feature verification.
4. Explicit persistence to models/ebnerd_reranker.joblib and models/mind_reranker.joblib.
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
    CANONICAL_GBDT_PARAMS,
    CANONICAL_SEED,
    CANONICAL_TRAIN_SAMPLE_N,
    EBNERD_CHECKPOINT_PATH,
    MIND_CHECKPOINT_PATH,
    PROJECT_ROOT,
)
from src.common.paths import interim_dir, processed_dir
from src.feature_store.article_store import ArticleFeatureStore
from src.feature_store.behavioral_features import BehavioralFeatureExtractor
from src.feature_store.session_features import PositionBiasModel, SessionFeatureExtractor
from src.reranking.feature_pipeline import FEATURE_NAMES, ReRankFeaturePipeline
from src.reranking.train_reranker import GBDTReranker, build_training_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def train_and_save_canonical_reranker(
    dataset: str,
    train_sample: int = CANONICAL_TRAIN_SAMPLE_N,
    scale: str = "small",
) -> tuple[GBDTReranker, Path, dict[str, Any]]:
    """Train and persist canonical GBDT re-ranker checkpoint."""
    logger.info("=" * 70)
    logger.info("CANONICAL RETRAINING PROTOCOL: dataset=%s, scale=%s", dataset, scale)
    logger.info("Seed: %d | Train Sample: %d impressions", CANONICAL_SEED, train_sample)
    logger.info("Hyperparameters: %s", CANONICAL_GBDT_PARAMS)
    logger.info("=" * 70)

    i_dir = interim_dir(dataset, scale)
    p_dir = processed_dir(dataset, scale)
    beh_path = i_dir / "behaviors.parquet"
    art_path = p_dir / "article_features.parquet"

    if not beh_path.exists():
        raise FileNotFoundError(f"Interim behaviors missing at {beh_path}")
    if not art_path.exists():
        raise FileNotFoundError(f"Processed article features missing at {art_path}")

    # 1. Load data
    article_store = ArticleFeatureStore(dataset, processed_dir=p_dir, scale=scale)
    df_beh = pl.read_parquet(beh_path)

    if "split" in df_beh.columns:
        train_df = df_beh.filter(pl.col("split") == "train")
    else:
        n = len(df_beh)
        train_df = df_beh.slice(0, int(n * 0.8))

    logger.info("Loaded %d training split behaviors for %s", len(train_df), dataset)

    # 2. Train-only popularity and position priors
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

    # 3. Build training dataset with chronological session grouping
    t0 = time.time()
    X_train, y_train, groups_train = build_training_dataset(
        dataset=dataset,
        behaviors_df=train_df,
        feature_pipeline=feature_pipeline,
        sample_limit=train_sample,
    )
    logger.info("Feature extraction completed in %.2fs: X=%s, y=%s", time.time() - t0, X_train.shape, y_train.shape)

    # 4. Verify session features are non-zero
    clicks_idx = FEATURE_NAMES.index("session_clicks_so_far_log")
    dwell_idx = FEATURE_NAMES.index("session_dwell_time_log")
    nz_clicks = int(np.sum(X_train[:, clicks_idx] > 0))
    nz_dwell = int(np.sum(X_train[:, dwell_idx] > 0))
    total_pairs = len(X_train)

    logger.info(
        "Session-feature verification (%s): non-zero session_clicks=%d/%d (%.2f%%), non-zero session_dwell=%d/%d (%.2f%%)",
        dataset, nz_clicks, total_pairs, (nz_clicks / max(1, total_pairs)) * 100,
        nz_dwell, total_pairs, (nz_dwell / max(1, total_pairs)) * 100,
    )

    if dataset == "ebnerd":
        assert nz_clicks > 0 and nz_dwell > 0, (
            f"EB-NeRD GBDT training has 0 non-zero session features: clicks={nz_clicks}, dwell={nz_dwell}"
        )

    # 5. Fit LightGBM model
    reranker = GBDTReranker(
        model_type=CANONICAL_GBDT_PARAMS["model_type"],
        n_estimators=CANONICAL_GBDT_PARAMS["n_estimators"],
        learning_rate=CANONICAL_GBDT_PARAMS["learning_rate"],
        max_depth=CANONICAL_GBDT_PARAMS["max_depth"],
        num_leaves=CANONICAL_GBDT_PARAMS["num_leaves"],
        random_state=CANONICAL_SEED,
    )
    t_fit = time.time()
    reranker.fit(X_train, y_train, groups=groups_train)
    logger.info("Model fitted in %.2fs", time.time() - t_fit)

    # 6. Save checkpoint
    out_path = EBNERD_CHECKPOINT_PATH if dataset == "ebnerd" else MIND_CHECKPOINT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reranker.save(out_path)
    logger.info("Saved canonical checkpoint to %s", out_path)

    # Confirm reloadable
    loaded = GBDTReranker.load(out_path)
    assert loaded.model is not None, "Reloaded model has None model object"
    logger.info("Successfully reloaded and verified model checkpoint from %s", out_path)

    meta = {
        "dataset": dataset,
        "checkpoint_path": str(out_path),
        "seed": CANONICAL_SEED,
        "n_train_impressions": train_sample,
        "n_train_candidate_pairs": total_pairs,
        "n_positive_clicks": int(y_train.sum()),
        "n_groups": len(groups_train),
        "non_zero_session_clicks": nz_clicks,
        "non_zero_session_dwell": nz_dwell,
        "hyperparameters": CANONICAL_GBDT_PARAMS,
        "top_feature_importances": list(reranker.get_feature_importances().items())[:10],
    }
    return reranker, out_path, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train canonical GBDT re-ranker")
    parser.add_argument("--dataset", choices=["ebnerd", "mind", "both"], default="ebnerd")
    parser.add_argument("--train-sample", type=int, default=CANONICAL_TRAIN_SAMPLE_N)
    args = parser.parse_args()

    datasets = ["ebnerd", "mind"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        train_and_save_canonical_reranker(ds, train_sample=args.train_sample)
