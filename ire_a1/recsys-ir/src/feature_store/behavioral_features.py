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
from dataclasses import dataclass, field
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
    recent_titles: list[str] = field(default_factory=list)
    recent_categories: list[str] = field(default_factory=list)
    recent_embeddings: np.ndarray | None = None
    user_embedding: np.ndarray | None = None

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
            "recent_titles": self.recent_titles,
            "recent_categories": self.recent_categories,
            "has_recent_embeddings": self.recent_embeddings is not None and len(self.recent_embeddings) > 0,
            "has_user_embedding": self.user_embedding is not None,
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
    user_history_embedding_similarity: float = 0.0
    user_history_max_embedding_sim: float = 0.0
    user_history_title_overlap: float = 0.0

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
            "user_history_embedding_similarity": self.user_history_embedding_similarity,
            "user_history_max_embedding_sim": self.user_history_max_embedding_sim,
            "user_history_title_overlap": self.user_history_title_overlap,
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
        article_index: Any | None = None,
    ) -> None:
        self.dataset = dataset.lower()
        self.article_store = article_store
        self.user_store = user_store
        self.train_popularity = train_popularity or {}
        self.train_inviews = train_inviews or {}
        self.history_cap = history_cap
        self.article_index = article_index

    def _get_embedding(self, article_id: str) -> np.ndarray | None:
        """Retrieve embedding vector for an article ID from index, dict, or store."""
        if self.article_index is None:
            return None
        aid = str(article_id)
        if hasattr(self.article_index, "get_embedding"):
            emb = self.article_index.get_embedding(aid)
            if emb is not None:
                return np.asarray(emb, dtype=np.float32)
        elif isinstance(self.article_index, dict):
            if aid in self.article_index:
                return np.asarray(self.article_index[aid], dtype=np.float32)
        elif callable(self.article_index):
            emb = self.article_index(aid)
            if emb is not None:
                return np.asarray(emb, dtype=np.float32)
        return None

    def _get_embeddings_batch(
        self, article_ids: list[str]
    ) -> tuple[dict[str, np.ndarray], int | None]:
        """Retrieve embedding vectors for a batch of article IDs."""
        if self.article_index is None or not article_ids:
            return {}, None
        str_ids = [str(x) for x in article_ids]
        if hasattr(self.article_index, "get_embeddings_batch"):
            mat, found_ids = self.article_index.get_embeddings_batch(str_ids)
            if len(found_ids) > 0 and mat.shape[0] > 0:
                dim = mat.shape[1]
                return {aid: mat[i].astype(np.float32) for i, aid in enumerate(found_ids)}, dim
            return {}, None
        elif isinstance(self.article_index, dict):
            found = {
                aid: np.asarray(self.article_index[aid], dtype=np.float32)
                for aid in str_ids
                if aid in self.article_index
            }
            dim = next(iter(found.values())).shape[0] if found else None
            return found, dim
        else:
            found = {}
            dim = None
            for aid in str_ids:
                emb = self._get_embedding(aid)
                if emb is not None:
                    found[aid] = emb
                    if dim is None:
                        dim = emb.shape[0]
            return found, dim

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

        # 3. Aggregate categories, subcategories, titles, and embeddings
        recent_ids = [str(entry["article_id"]) for entry in recent]
        article_rows = self.article_store.get_articles_batch(recent_ids)
        art_map = {str(row["article_id"]): row for row in article_rows}

        recent_titles: list[str] = []
        recent_categories: list[str] = []
        cat_weights: Counter[str] = Counter()
        subcat_weights: Counter[str] = Counter()

        for idx, aid in enumerate(recent_ids):
            w = weights[idx] if idx < len(weights) else 1.0
            art = art_map.get(aid, {})
            title = str(art.get("title") or art.get("cleaned_text") or "")
            recent_titles.append(title)
            cat = str(art.get("category") or "")
            recent_categories.append(cat)
            if cat:
                cat_weights[cat] += w
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

        # 4. Extract recent embeddings and recency-weighted pooled user profile embedding
        emb_map, dim = self._get_embeddings_batch(recent_ids)
        recent_embeddings_list: list[np.ndarray] = []
        user_vector = np.zeros(dim, dtype=np.float32) if dim else None
        valid_emb_count = 0

        for idx, aid in enumerate(recent_ids):
            w = weights[idx] if idx < len(weights) else 1.0
            if aid in emb_map:
                emb = emb_map[aid]
                recent_embeddings_list.append(emb)
                if user_vector is not None:
                    user_vector += w * emb
                    valid_emb_count += 1

        if valid_emb_count > 0 and user_vector is not None:
            norm = np.linalg.norm(user_vector)
            if norm > 0:
                user_vector = user_vector / norm
            user_embedding = user_vector.astype(np.float32)
        else:
            user_embedding = None

        recent_embeddings = (
            np.array(recent_embeddings_list, dtype=np.float32)
            if recent_embeddings_list
            else None
        )

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
            recent_titles=recent_titles,
            recent_categories=recent_categories,
            recent_embeddings=recent_embeddings,
            user_embedding=user_embedding,
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
        - User click-history embedding cosine similarity and max item similarity
        - Title lexical overlap (Jaccard) with user's recent clicked titles

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
        meta_rows = self.article_store.get_articles_batch(cands)
        meta_map = {str(row["article_id"]): row for row in meta_rows}

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

            # Title word overlap (Jaccard similarity) with user's recent clicked titles
            cand_title = str(meta.get("title") or meta.get("cleaned_text") or "")
            title_overlap = 0.0
            if cand_title and user_summary.recent_titles:
                cand_words = set(cand_title.lower().split())
                hist_words = set()
                for ht in user_summary.recent_titles:
                    if ht:
                        hist_words.update(ht.lower().split())
                if cand_words and hist_words:
                    intersection = cand_words & hist_words
                    union = cand_words | hist_words
                    title_overlap = float(len(intersection) / len(union)) if union else 0.0

            # History embedding similarities
            emb_sim = 0.0
            max_emb_sim = 0.0
            cand_emb = self._get_embedding(cid)
            if cand_emb is not None:
                cand_norm = np.linalg.norm(cand_emb)
                c_unit = cand_emb / cand_norm if cand_norm > 0 else None
                if c_unit is not None:
                    if user_summary.user_embedding is not None:
                        emb_sim = float(np.dot(user_summary.user_embedding, c_unit))
                        emb_sim = max(-1.0, min(1.0, emb_sim))
                    if user_summary.recent_embeddings is not None and len(user_summary.recent_embeddings) > 0:
                        rec_norms = np.linalg.norm(user_summary.recent_embeddings, axis=1, keepdims=True)
                        rec_norms = np.where(rec_norms == 0, 1.0, rec_norms)
                        normed_rec = user_summary.recent_embeddings / rec_norms
                        sims = np.dot(normed_rec, c_unit)
                        max_emb_sim = float(np.max(sims))
                        max_emb_sim = max(-1.0, min(1.0, max_emb_sim))

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
                    user_history_embedding_similarity=emb_sim,
                    user_history_max_embedding_sim=max_emb_sim,
                    user_history_title_overlap=title_overlap,
                )
            )

        return results
