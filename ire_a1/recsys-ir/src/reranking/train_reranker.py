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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract candidate-level training rows (X, y, groups) from behaviors DataFrame."""
    X_rows, y_rows, groups = [], [], []

    rows = behaviors_df.iter_rows(named=True)
    gid = 0
    count = 0

    for row in rows:
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

        ts = row.get("timestamp")
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

        X_imp = feature_pipeline.extract_impression_features(
            user_id=user_id,
            as_of_ts=as_of_ts,
            candidate_ids=cands,
            user_history=user_history,
            current_session_id=session_id,
        )

        for i in range(len(cands)):
            X_rows.append(X_imp[i])
            y_rows.append(labels[i])
            groups.append(gid)

        gid += 1
        count += 1

    if not X_rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int64), np.array(groups, dtype=np.int64)
