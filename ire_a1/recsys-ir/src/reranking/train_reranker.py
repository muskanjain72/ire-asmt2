"""Train a Gradient-Boosted Decision Tree (GBDT) Re-Ranker.

Trains a LightGBM or HistGradientBoosting model over the tabular features
engineered from Question 1 (session, dwell time, position bias, article freshness,
train popularity, category affinity, and 1st-stage retrieval scores).
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Any, Sequence

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from src.feature_store.article_store import ArticleFeatureStore
from src.feature_store.behavioral_features import BehavioralFeatureExtractor
from src.feature_store.session_features import PositionBiasModel, SessionFeatureExtractor
from src.reranking.feature_pipeline import FEATURE_NAMES, ReRankFeaturePipeline

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _PROJECT_ROOT / "models"


class GBDTReranker:
    """Gradient Boosted Decision Tree re-ranker with group ranking awareness."""

    def __init__(
        self,
        model_type: str = "lightgbm",
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        random_state: int = 42,
    ) -> None:
        self.model_type = "lightgbm" if (model_type == "lightgbm" and HAS_LIGHTGBM) else "hist_gbdt"
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.model = None
        self.feature_names = list(FEATURE_NAMES)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None = None,
    ) -> GBDTReranker:
        """Fit re-ranker on candidate feature matrix X and click labels y."""
        logger.info(
            "Fitting %s re-ranker on %d candidate pairs (%d positive)",
            self.model_type, len(y), int(y.sum()),
        )

        if self.model_type == "lightgbm" and HAS_LIGHTGBM:
            self.model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                num_leaves=self.num_leaves,
                random_state=self.random_state,
                verbosity=-1,
                force_col_wise=True,
            )
            self.model.fit(X, y)
        else:
            self.model = HistGradientBoostingClassifier(
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                max_leaf_nodes=self.num_leaves,
                random_state=self.random_state,
            )
            self.model.fit(X, y)

        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Predict click probabilities P(click = 1 | X)."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if len(X) == 0:
            return np.empty((0,), dtype=np.float64)
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self) -> dict[str, float]:
        """Return dict of {feature_name: importance}, sorted descending."""
        if self.model is None:
            return {}
        if hasattr(self.model, "feature_importances_"):
            raw_imp = self.model.feature_importances_
            total = max(1e-12, float(raw_imp.sum()))
            norm_imp = [float(v / total) for v in raw_imp]
            mapping = dict(zip(self.feature_names, norm_imp))
        else:
            # Equal fallback if model does not expose feature importances
            mapping = {name: 1.0 / len(self.feature_names) for name in self.feature_names}
        return dict(sorted(mapping.items(), key=lambda x: x[1], reverse=True))

    def save(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "model_type": self.model_type,
                "feature_names": self.feature_names,
                "hyperparams": {
                    "n_estimators": self.n_estimators,
                    "learning_rate": self.learning_rate,
                    "max_depth": self.max_depth,
                    "num_leaves": self.num_leaves,
                },
            },
            output_path,
        )
        logger.info("Saved re-ranker model to %s", output_path)
        return output_path

    @classmethod
    def load(cls, model_path: Path) -> GBDTReranker:
        data = joblib.load(model_path)
        reranker = cls(model_type=data["model_type"], **data.get("hyperparams", {}))
        reranker.model = data["model"]
        reranker.feature_names = data.get("feature_names", list(FEATURE_NAMES))
        return reranker


def build_training_dataset(
    dataset: str,
    behaviors_df: pl.DataFrame,
    feature_pipeline: ReRankFeaturePipeline,
    sample_limit: int | None = None,
    embed_scores_map: dict[str, dict[str, float]] | None = None,
    bm25_scores_map: dict[str, dict[str, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract candidate-level training rows (X, y, groups) from behaviors DataFrame.

    Chronologically processes each user's impressions, accumulating prior impressions
    within each session to faithfully compute session-level features without data leakage.
    If embed_scores_map or bm25_scores_map is provided, populates the Stage-1 retrieval
    signals (retrieval_bm25_score, retrieval_embed_sim, retrieval_hybrid_score).
    """
    X_rows, y_rows, groups = [], [], []

    user_col = "user_id" if "user_id" in behaviors_df.columns else None
    ts_col = "timestamp" if "timestamp" in behaviors_df.columns else ("impression_time" if "impression_time" in behaviors_df.columns else None)

    if user_col and ts_col:
        df_sorted = behaviors_df.sort([user_col, ts_col])
    elif user_col:
        df_sorted = behaviors_df.sort(user_col)
    elif ts_col:
        df_sorted = behaviors_df.sort(ts_col)
    else:
        df_sorted = behaviors_df

    user_prior_impressions: dict[str, list[dict[str, Any]]] = {}
    gid = 0
    count = 0

    for row in df_sorted.iter_rows(named=True):
        if sample_limit is not None and count >= sample_limit:
            break

        raw_cands = row.get("candidates")
        raw_labels = row.get("labels")
        if not raw_cands or not raw_labels:
            continue

        cands = json.loads(raw_cands) if isinstance(raw_cands, str) else list(raw_cands)
        labels = json.loads(raw_labels) if isinstance(raw_labels, str) else list(raw_labels)

        # Skip non-informative impressions (no positive or all positive)
        pos_count = sum(labels)
        if pos_count == 0 or pos_count == len(labels):
            continue

        ts = row.get("timestamp") or row.get("impression_time")
        if isinstance(ts, str):
            as_of_ts = datetime.fromisoformat(ts)
        elif isinstance(ts, datetime):
            as_of_ts = ts
        else:
            as_of_ts = datetime(2023, 5, 20, 12, 0, 0)

        history_raw = row.get("clicked_history")
        user_history = (
            json.loads(history_raw) if isinstance(history_raw, str) else (history_raw or [])
        )
        if user_history and isinstance(user_history[0], str):
            user_history = [{"article_id": aid, "clicked_at": None} for aid in user_history]

        user_id = str(row.get("user_id", "U_UNKNOWN"))
        session_id = row.get("session_id")
        prior_imps = user_prior_impressions.get(user_id, [])

        imp_id = str(row.get("impression_id", ""))
        e_scores = embed_scores_map.get(imp_id) if embed_scores_map else None
        b_scores = bm25_scores_map.get(imp_id) if bm25_scores_map else None

        X_imp = feature_pipeline.extract_impression_features(
            user_id=user_id,
            as_of_ts=as_of_ts,
            candidate_ids=cands,
            user_history=user_history,
            user_impressions=prior_imps,
            current_session_id=session_id,
            embed_scores=e_scores,
            bm25_scores=b_scores,
        )

        for i in range(len(cands)):
            X_rows.append(X_imp[i])
            y_rows.append(labels[i])
            groups.append(gid)

        gid += 1
        count += 1

        # Record this impression into user's prior impression timeline
        if user_id not in user_prior_impressions:
            user_prior_impressions[user_id] = []
        user_prior_impressions[user_id].append({
            "timestamp": as_of_ts,
            "session_id": session_id,
            "labels": labels,
            "read_time": row.get("read_time"),
            "scroll_percentage": row.get("scroll_percentage"),
        })

    if not X_rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    X_arr = np.array(X_rows, dtype=np.float32)
    y_arr = np.array(y_rows, dtype=np.int64)
    groups_arr = np.array(groups, dtype=np.int64)

    # Sanity check: confirm session features are non-zero for at least some training rows
    clicks_idx = FEATURE_NAMES.index("session_clicks_so_far_log")
    dwell_idx = FEATURE_NAMES.index("session_dwell_time_log")
    nz_clicks = int(np.sum(X_arr[:, clicks_idx] > 0))
    nz_dwell = int(np.sum(X_arr[:, dwell_idx] > 0))
    logger.info(
        "[%s] Session-feature verification: non-zero session_clicks_so_far=%d/%d, non-zero session_dwell_time_so_far=%d/%d",
        dataset, nz_clicks, len(X_arr), nz_dwell, len(X_arr),
    )
    print(
        f"[SESSION FEATURE CHECK] dataset={dataset}: non-zero session_clicks_so_far={nz_clicks}/{len(X_arr)}, non-zero session_dwell_time_so_far={nz_dwell}/{len(X_arr)}"
    )
    if len(X_arr) >= 200:
        if dataset.lower() == "ebnerd":
            assert nz_clicks > 0 or nz_dwell > 0, (
                f"Expected non-zero session clicks or dwell time for {dataset}, got clicks={nz_clicks}, dwell={nz_dwell}"
            )
        elif dataset.lower() == "mind":
            # In MIND, raw behaviors lack session_id and dwell time (read_time/scroll_percentage).
            # Furthermore, in the interim behaviors slice, 98.97% of users have exactly 1 impression,
            # so prior session impressions evaluate to 0.0 by dataset design.
            logger.info("[%s] Note: session features are 0.0 due to absence of session_id/dwell in MIND schema.", dataset)

    return X_arr, y_arr, groups_arr
