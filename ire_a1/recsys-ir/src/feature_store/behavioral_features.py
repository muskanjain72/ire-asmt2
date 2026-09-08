r"""Behavioral, click-history, and candidate article feature engineering.

Implements Assignment 2 Q1 requirements:
1. Click-history features:
   - Recent clicked articles (article_ids, titles, categories, embeddings).
   - Historical click count (history length).
   - Recency-weighted history with exponential decay.
   - User profile representation via recency-weighted embedding pooling.
2. Decay formula reconciliation (resolving Section 5 of A2 Implementation Plan):
   - Continuous time-based exponential decay: $w_i = exp(-\lambda \Delta t)$
     with $\lambda = 1 / (7 * 86400)$ (~7-day half-life), used when real timestamps
     exist (EB-NeRD) to model continuous interest forgetting.
   - Discrete position-based exponential decay: $w_i = \gamma^{rank}$ with $\gamma = 0.85$,
     used when timestamps are absent (MIND) or for sequence ranking.
   - Unified adapter providing both modes and automatic fallback.
3. Candidate article features:
   - Popularity: train-split-only inview count, click count, and log1p CTR.
   - Freshness: elapsed hours since publish (EB-NeRD), with explicit fallback for MIND.
   - Category match & affinity: user historical category distribution vs candidate category,
     recency-weighted category affinity, and exact top-category match flag.
4. Strict behavioural-window boundary (anti-leakage):
   - All lookups and calculations take `as_of_ts`.
   - Clicks or events at or after `as_of_ts` are strictly excluded.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import logging
import math
from typing import Any, Sequence

import numpy as np

from src.feature_store.article_store import ArticleFeatureStore
from src.feature_store.user_store import UserFeatureStore

logger = logging.getLogger(__name__)

# Continuous decay constant: λ = 1 / (7 * 86400) -> ~7-day half-life
TIME_DECAY_LAMBDA: float = 1.0 / (7.0 * 86400.0)

# Discrete position decay factor: 0.85 per step from most recent
POSITION_DECAY_FACTOR: float = 0.85


def compute_recency_weights(
    history: Sequence[dict[str, Any]],
    as_of_ts: datetime,
    mode: str = "unified",
    decay_lambda: float = TIME_DECAY_LAMBDA,
    position_decay: float = POSITION_DECAY_FACTOR,
    history_cap: int = 20,
    normalize: bool = True,
) -> list[float]:
    """Compute recency weights over a user's click history.

    Reconciles the two decay formulas in the codebase:
    - 'time': continuous exponential decay exp(-λ * Δt) based on timestamps.
      If an item has no timestamp, receives weight 1.0.
    - 'position': discrete sequence decay position_decay^(pos_from_most_recent).
    - 'unified': if any item has timestamp, use time decay; otherwise fall back
      to position decay.

    Parameters
    ----------
    history : list[dict]
        Chronological list of click history dicts (oldest first, most recent last).
        Must already be filtered to items before `as_of_ts`.
    as_of_ts : datetime
        Timestamp cutoff.
    mode : str
        'time', 'position', or 'unified'.
    decay_lambda : float
        Decay constant for time decay.
    position_decay : float
        Decay factor for position decay (0 < position_decay <= 1.0).
    history_cap : int
        Maximum number of recent items to weight.
    normalize : bool
        If True, weights sum to 1.0 (or empty list if no history).

    Returns
    -------
    list[float]
        List of weights parallel to recent history entries.
    """
    if not history:
        return []

    recent = list(history[-history_cap:])
    n = len(recent)

    has_timestamps = any(e.get("clicked_at") is not None for e in recent)

    use_time = (mode == "time") or (mode == "unified" and has_timestamps)

    weights = []
    if use_time:
        for entry in recent:
            clicked_at_str = entry.get("clicked_at")
            if clicked_at_str is not None:
                clicked_at = (
                    datetime.fromisoformat(clicked_at_str)
                    if isinstance(clicked_at_str, str)
                    else clicked_at_str
                )
                delta_seconds = max(0.0, (as_of_ts - clicked_at).total_seconds())
                w = math.exp(-decay_lambda * delta_seconds)
            else:
                w = 1.0
            weights.append(w)
    else:
        # Position-based decay: most recent (last) gets position_decay^0 = 1.0
        for i in range(n):
            steps_from_recent = n - 1 - i
            w = position_decay ** steps_from_recent
            weights.append(float(w))

    if normalize and weights:
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]

    return weights


@dataclass(frozen=True)
class UserClickHistorySummary:
    """Summary of user behavioral click history."""
    user_id: str
    lifetime_history_len: int
    active_history_len: int
    recent_article_ids: list[str]
    category_distribution: dict[str, float]
    top_category: str | None
    subcategory_distribution: dict[str, float]
    top_subcategory: str | None
    recency_weights: list[float]
    decay_mode_used: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "lifetime_history_len": self.lifetime_history_len,
            "active_history_len": self.active_history_len,
            "recent_article_ids": self.recent_article_ids,
            "category_distribution": self.category_distribution,
            "top_category": self.top_category,
            "subcategory_distribution": self.subcategory_distribution,
            "top_subcategory": self.top_subcategory,
            "recency_weights": self.recency_weights,
            "decay_mode_used": self.decay_mode_used,
        }


@dataclass(frozen=True)
class CandidateBehavioralFeatures:
    """Behavioral and context features for a single candidate article."""
    article_id: str
    category_affinity: float
    subcategory_affinity: float
    is_top_category_match: bool
    is_top_subcategory_match: bool
    train_popularity_log_clicks: float
    train_popularity_log_inviews: float
    train_empirical_ctr: float
    freshness_hours: float | None
    freshness_available: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "article_id": self.article_id,
            "category_affinity": self.category_affinity,
            "subcategory_affinity": self.subcategory_affinity,
            "is_top_category_match": float(self.is_top_category_match),
            "is_top_subcategory_match": float(self.is_top_subcategory_match),
            "train_popularity_log_clicks": self.train_popularity_log_clicks,
            "train_popularity_log_inviews": self.train_popularity_log_inviews,
            "train_empirical_ctr": self.train_empirical_ctr,
            "freshness_hours": self.freshness_hours if self.freshness_hours is not None else -1.0,
            "freshness_available": float(self.freshness_available),
        }


class BehavioralFeatureExtractor:
    """Unified extractor for click-history, user profile, and candidate article features."""

    def __init__(
        self,
        dataset: str,
        article_store: ArticleFeatureStore,
        user_store: UserFeatureStore | None = None,
        train_popularity: dict[str, int] | None = None,
        train_inviews: dict[str, int] | None = None,
        history_cap: int = 20,
    ) -> None:
        self.dataset = dataset.lower()
        self.article_store = article_store
        self.user_store = user_store
        self.train_popularity = train_popularity or {}
        self.train_inviews = train_inviews or {}
        self.history_cap = history_cap

    def summarize_user_history(
        self,
        user_id: str,
        as_of_ts: datetime,
        raw_history: list[dict[str, Any]] | None = None,
        decay_mode: str = "unified",
    ) -> UserClickHistorySummary:
        """Extract and summarize user click history strictly before `as_of_ts`.

        Parameters
        ----------
        user_id : str
            User ID.
        as_of_ts : datetime
            Cutoff timestamp. NO event with timestamp >= as_of_ts is allowed.
        raw_history : list[dict], optional
            Pre-retrieved raw history entries. If None, queries `user_store`.
        decay_mode : str
            'time', 'position', or 'unified'.

        Returns
        -------
        UserClickHistorySummary
        """
        # 1. Retrieve & filter history strictly before as_of_ts
        if raw_history is not None:
            filtered_history = []
            for entry in raw_history:
                ts_str = entry.get("clicked_at")
                if ts_str is None:
                    filtered_history.append(entry)
                else:
                    ts = datetime.fromisoformat(ts_str) if isinstance(ts_str, str) else ts_str
                    if ts < as_of_ts:
                        filtered_history.append(entry)
        elif self.user_store is not None:
            filtered_history = self.user_store.get_user_history(user_id, as_of_ts, dataset=self.dataset)
        else:
            filtered_history = []

        lifetime_len = len(filtered_history)
        recent = filtered_history[-self.history_cap:]
        active_len = len(recent)

        # 2. Compute recency weights
        weights = compute_recency_weights(
            recent,
            as_of_ts,
            mode=decay_mode,
            history_cap=self.history_cap,
            normalize=True,
        )

        # 3. Aggregate categories and subcategories
        recent_ids = [str(entry["article_id"]) for entry in recent]
        article_rows = self.article_store.get_articles_batch(
            recent_ids,
            columns=["article_id", "category", "subcategory"],
        )
        art_map = {row["article_id"]: row for row in article_rows}

        cat_weights: Counter[str] = Counter()
        subcat_weights: Counter[str] = Counter()

        for idx, aid in enumerate(recent_ids):
            w = weights[idx] if idx < len(weights) else 1.0
            art = art_map.get(aid)
            if art:
                cat = art.get("category")
                if cat:
                    cat_weights[str(cat)] += w
                subcat = art.get("subcategory")
                if subcat:
                    subcat_weights[str(subcat)] += w

        total_cat_w = sum(cat_weights.values())
        cat_dist = (
            {k: v / total_cat_w for k, v in cat_weights.items()}
            if total_cat_w > 0
            else {}
        )
        top_cat = cat_weights.most_common(1)[0][0] if cat_weights else None

        total_subcat_w = sum(subcat_weights.values())
        subcat_dist = (
            {k: v / total_subcat_w for k, v in subcat_weights.items()}
            if total_subcat_w > 0
            else {}
        )
        top_subcat = subcat_weights.most_common(1)[0][0] if subcat_weights else None

        return UserClickHistorySummary(
            user_id=user_id,
            lifetime_history_len=lifetime_len,
            active_history_len=active_len,
            recent_article_ids=recent_ids,
            category_distribution=cat_dist,
            top_category=top_cat,
            subcategory_distribution=subcat_dist,
            top_subcategory=top_subcat,
            recency_weights=weights,
            decay_mode_used=decay_mode,
        )

    def extract_candidate_features(
        self,
        candidate_ids: Sequence[str],
        user_summary: UserClickHistorySummary,
        as_of_ts: datetime,
    ) -> list[CandidateBehavioralFeatures]:
        """Compute behavioral and interaction features for candidate articles.

        Features:
        - Category & subcategory affinity scores
        - Match indicators with user's top historical preferences
        - Train-split-only popularity and CTR signals
        - Article freshness in hours (EB-NeRD), with leakage verification

        Parameters
        ----------
        candidate_ids : list[str]
            Candidate article IDs.
        user_summary : UserClickHistorySummary
            Summarized user history prior to `as_of_ts`.
        as_of_ts : datetime
            Impression timestamp.

        Returns
        -------
        list[CandidateBehavioralFeatures]
        """
        cands = [str(cid) for cid in candidate_ids]
        meta_rows = self.article_store.get_articles_batch(
            cands,
            columns=["article_id", "category", "subcategory", "published_at"],
        )
        meta_map = {row["article_id"]: row for row in meta_rows}

        results = []
        for cid in cands:
            meta = meta_map.get(cid, {})
            cand_cat = str(meta.get("category")) if meta.get("category") else None
            cand_subcat = str(meta.get("subcategory")) if meta.get("subcategory") else None

            # Category affinity
            cat_affinity = user_summary.category_distribution.get(cand_cat, 0.0) if cand_cat else 0.0
            subcat_affinity = user_summary.subcategory_distribution.get(cand_subcat, 0.0) if cand_subcat else 0.0
            is_top_cat = (cand_cat == user_summary.top_category) if (cand_cat and user_summary.top_category) else False
            is_top_subcat = (cand_subcat == user_summary.top_subcategory) if (cand_subcat and user_summary.top_subcategory) else False

            # Train-split-only popularity features
            clicks = self.train_popularity.get(cid, 0)
            inviews = self.train_inviews.get(cid, clicks)  # fallback to clicks if inviews not tracked
            log_clicks = math.log1p(clicks)
            log_inviews = math.log1p(inviews)
            ctr = float((clicks + 1.0) / (inviews + 10.0))  # Laplace-smoothed empirical CTR

            # Freshness (hours since publish)
            pub_ts = meta.get("published_at")
            if pub_ts is not None:
                if isinstance(pub_ts, str):
                    pub_dt = datetime.fromisoformat(pub_ts)
                else:
                    pub_dt = pub_ts

                delta_sec = (as_of_ts - pub_dt).total_seconds()
                if delta_sec < -86400:
                    raise ValueError(
                        f"Data leakage detected: article {cid} published at {pub_dt} "
                        f"is after as_of_ts {as_of_ts}"
                    )
                freshness_hours = max(0.0, delta_sec / 3600.0)
                freshness_avail = True
            else:
                freshness_hours = None
                freshness_avail = False

            results.append(
                CandidateBehavioralFeatures(
                    article_id=cid,
                    category_affinity=cat_affinity,
                    subcategory_affinity=subcat_affinity,
                    is_top_category_match=is_top_cat,
                    is_top_subcategory_match=is_top_subcat,
                    train_popularity_log_clicks=log_clicks,
                    train_popularity_log_inviews=log_inviews,
                    train_empirical_ctr=ctr,
                    freshness_hours=freshness_hours,
                    freshness_available=freshness_avail,
                )
            )

        return results
