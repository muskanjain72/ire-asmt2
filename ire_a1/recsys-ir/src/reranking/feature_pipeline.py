"""Feature pipeline for two-stage re-ranking.

Vectorizes tabular features for each candidate in an impression by combining:
1. Stage-1 retrieval signals (BM25, embedding cosine similarity, hybrid blend, rank).
2. Q1 click-history signals (lifetime/active counts, recency weights).
3. Q1 session dynamics (session impression index, session clicks, dwell time, scroll percentage, session age).
4. Q1 display position bias (rank, reciprocal rank, log discount, empirical CTR prior).
5. Q1 candidate article signals (train popularity, empirical CTR, freshness, category affinity).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import logging
import math
from typing import Any, Sequence

import numpy as np

from src.feature_store.article_store import ArticleFeatureStore
from src.feature_store.behavioral_features import (
    BehavioralFeatureExtractor,
    UserClickHistorySummary,
)
from src.feature_store.session_features import (
    PositionBiasFeatures,
    PositionBiasModel,
    SessionContext,
    SessionFeatureExtractor,
)

logger = logging.getLogger(__name__)

FEATURE_NAMES: list[str] = [
    # 1. Stage-1 Retrieval signals (4)
    "retrieval_bm25_score",
    "retrieval_embed_sim",
    "retrieval_hybrid_score",
    "retrieval_rank_pct",
    # 2. Q1 Click-History features (6)
    "user_lifetime_history_log",
    "user_active_history_log",
    "user_mean_recency_weight",
    "user_history_embedding_similarity",
    "user_history_max_embedding_sim",
    "user_history_title_overlap",
    # 3. Q1 Session features (7)
    "session_impression_index",
    "session_clicks_so_far_log",
    "session_dwell_time_log",
    "session_mean_scroll",
    "session_time_since_start_hours",
    "session_time_since_last_min",
    "session_dwell_available",
    # 4. Q1 Position bias features (5)
    "position_bias_rank",
    "position_bias_relative",
    "position_bias_reciprocal",
    "position_bias_log_discount",
    "position_bias_empirical_ctr",
    # 5. Q1 Article & Interaction features (9)
    "article_train_pop_clicks_log",
    "article_train_pop_inviews_log",
    "article_train_empirical_ctr",
    "article_freshness_hours",
    "article_freshness_available",
    "article_category_affinity",
    "article_subcategory_affinity",
    "article_is_top_category_match",
    "article_is_top_subcategory_match",
]


class ReRankFeaturePipeline:
    """Assembles unified tabular feature matrices for re-ranking candidates."""

    def __init__(
        self,
        dataset: str,
        article_store: ArticleFeatureStore,
        session_extractor: SessionFeatureExtractor | None = None,
        behavioral_extractor: BehavioralFeatureExtractor | None = None,
        article_index: Any | None = None,
    ) -> None:
        self.dataset = dataset.lower()
        self.article_store = article_store
        self.session_extractor = session_extractor or SessionFeatureExtractor(dataset=self.dataset)
        self.behavioral_extractor = behavioral_extractor or BehavioralFeatureExtractor(
            dataset=self.dataset, article_store=article_store, article_index=article_index
        )
        self.feature_names = list(FEATURE_NAMES)

    def extract_impression_features(
        self,
        user_id: str,
        as_of_ts: datetime,
        candidate_ids: Sequence[str],
        user_history: list[dict[str, Any]] | None = None,
        user_impressions: list[dict[str, Any]] | None = None,
        current_session_id: str | int | None = None,
        bm25_scores: dict[str, float] | None = None,
        embed_scores: dict[str, float] | None = None,
        hybrid_scores: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Extract a 2D feature matrix of shape (num_candidates, num_features).

        Parameters
        ----------
        user_id : str
            User identifier.
        as_of_ts : datetime
            Cutoff timestamp for strict anti-leakage boundary.
        candidate_ids : Sequence[str]
            Ordered list of candidates to score (e.g. from Stage 1 retrieval).
        user_history : list[dict], optional
            Past click history entries.
        user_impressions : list[dict], optional
            Past impression events for this user (for session extraction).
        current_session_id : str or int, optional
            Explicit session identifier if known (EB-NeRD).
        bm25_scores : dict[str, float], optional
            Mapping of candidate_id -> stage 1 BM25 score.
        embed_scores : dict[str, float], optional
            Mapping of candidate_id -> stage 1 embedding similarity.
        hybrid_scores : dict[str, float], optional
            Mapping of candidate_id -> stage 1 hybrid score.

        Returns
        -------
        np.ndarray
            Matrix of shape (len(candidate_ids), len(FEATURE_NAMES)), dtype float32.
        """
        cands = [str(cid) for cid in candidate_ids]
        K = len(cands)
        if K == 0:
            return np.empty((0, len(self.feature_names)), dtype=np.float32)

        bm25_map = bm25_scores or {}
        embed_map = embed_scores or {}
        hybrid_map = hybrid_scores or {}

        # 1. User Click-History Summary (strictly before as_of_ts)
        u_summary = self.behavioral_extractor.summarize_user_history(
            user_id=user_id,
            as_of_ts=as_of_ts,
            raw_history=user_history,
            decay_mode="unified",
        )

        # 2. Session Context (strictly before as_of_ts)
        s_ctx = self.session_extractor.extract_session_context(
            user_id=user_id,
            as_of_ts=as_of_ts,
            user_impressions=user_impressions or [],
            current_session_id=current_session_id,
        )

        # 3. Position Bias signals for candidates
        pos_features = self.session_extractor.extract_candidate_position_features(cands)

        # 4. Candidate Behavioral & Article signals
        cand_behavioral = self.behavioral_extractor.extract_candidate_features(
            candidate_ids=cands,
            user_summary=u_summary,
            as_of_ts=as_of_ts,
        )

        # 5. Build tabular rows
        rows = []
        u_life_log = math.log1p(u_summary.lifetime_history_len)
        u_act_log = math.log1p(u_summary.active_history_len)
        u_mean_w = float(np.mean(u_summary.recency_weights)) if u_summary.recency_weights else 0.0

        s_imp_idx = float(s_ctx.impression_index_in_session)
        s_clicks_log = math.log1p(s_ctx.clicks_in_session_before_now)
        s_dwell_log = math.log1p(max(0.0, s_ctx.dwell_time_in_session_before_now))
        s_scroll = float(s_ctx.mean_scroll_percentage_before_now)
        s_time_start_h = float(s_ctx.time_since_session_start_seconds / 3600.0)
        s_time_last_m = (
            float(s_ctx.time_since_last_impression_seconds / 60.0)
            if s_ctx.time_since_last_impression_seconds >= 0.0
            else -1.0
        )
        s_dwell_avail = 1.0 if s_ctx.dwell_time_available else 0.0

        for i, cid in enumerate(cands):
            # Retrieval features
            b_val = float(bm25_map.get(cid, 0.0))
            e_val = float(embed_map.get(cid, 0.0))
            h_val = float(hybrid_map.get(cid, (b_val + e_val) / 2.0))
            r_rank_pct = float(i / max(1, K - 1))

            # Position bias features
            p_feat = pos_features[i]
            p_rank = float(p_feat.position)
            p_rel = float(p_feat.relative_position)
            p_recip = float(p_feat.reciprocal_rank)
            p_log = float(p_feat.log_position_discount)
            p_ctr = float(p_feat.empirical_position_ctr)

            # Candidate article & interaction features
            c_feat = cand_behavioral[i]

            # Click-history embedding and title overlap signals
            hist_emb_sim = float(c_feat.user_history_embedding_similarity)
            if hist_emb_sim == 0.0 and cid in embed_map:
                hist_emb_sim = float(embed_map[cid])
            hist_max_sim = float(c_feat.user_history_max_embedding_sim)
            if hist_max_sim == 0.0 and hist_emb_sim != 0.0:
                hist_max_sim = hist_emb_sim
            hist_title_ov = float(c_feat.user_history_title_overlap)

            art_clicks_log = float(c_feat.train_popularity_log_clicks)
            art_inviews_log = float(c_feat.train_popularity_log_inviews)
            art_ctr = float(c_feat.train_empirical_ctr)
            art_fresh_h = float(c_feat.freshness_hours) if c_feat.freshness_hours is not None else -1.0
            art_fresh_avail = 1.0 if c_feat.freshness_available else 0.0
            art_cat_aff = float(c_feat.category_affinity)
            art_subcat_aff = float(c_feat.subcategory_affinity)
            art_top_cat = 1.0 if c_feat.is_top_category_match else 0.0
            art_top_subcat = 1.0 if c_feat.is_top_subcategory_match else 0.0

            rows.append([
                b_val, e_val, h_val, r_rank_pct,
                u_life_log, u_act_log, u_mean_w, hist_emb_sim, hist_max_sim, hist_title_ov,
                s_imp_idx, s_clicks_log, s_dwell_log, s_scroll, s_time_start_h, s_time_last_m, s_dwell_avail,
                p_rank, p_rel, p_recip, p_log, p_ctr,
                art_clicks_log, art_inviews_log, art_ctr, art_fresh_h, art_fresh_avail,
                art_cat_aff, art_subcat_aff, art_top_cat, art_top_subcat,
            ])

        return np.array(rows, dtype=np.float32)
