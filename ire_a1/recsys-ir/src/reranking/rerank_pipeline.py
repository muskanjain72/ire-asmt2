"""Two-stage retrieve-then-rank inference pipeline.

Connects Stage-1 Candidate Generation (Lexical / Semantic / Hybrid)
to Stage-2 Behavioral Re-Ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.evaluation.ranking_metrics import auc_score, mrr, ndcg_at_k
from src.reranking.feature_pipeline import ReRankFeaturePipeline
from src.reranking.train_reranker import GBDTReranker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReRankingResult:
    """Output of candidate re-ranking for a single impression."""
    original_candidate_ids: list[str]
    reranked_candidate_ids: list[str]
    reranked_scores: list[float]
    original_metrics: dict[str, float]
    reranked_metrics: dict[str, float]

    def metric_deltas(self) -> dict[str, float]:
        """Return delta (reranked - original) for each metric."""
        deltas = {}
        for m in self.original_metrics:
            if m in self.reranked_metrics:
                deltas[f"delta_{m}"] = self.reranked_metrics[m] - self.original_metrics[m]
        return deltas


class TwoStageRetrieveThenRank:
    """End-to-end retrieve-then-rank pipeline."""

    def __init__(
        self,
        feature_pipeline: ReRankFeaturePipeline,
        reranker: GBDTReranker,
    ) -> None:
        self.feature_pipeline = feature_pipeline
        self.reranker = reranker

    def rerank_candidates(
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
        labels: list[int] | None = None,
    ) -> ReRankingResult:
        """Score and re-rank an impression's candidates.

        Parameters
        ----------
        user_id : str
            User ID.
        as_of_ts : datetime
            Cutoff timestamp.
        candidate_ids : list[str]
            Retrieved candidates in Stage 1 rank order.
        user_history : list[dict], optional
            User history entries prior to `as_of_ts`.
        user_impressions : list[dict], optional
            User impression events for session features.
        current_session_id : str or int, optional
            Session ID if known.
        bm25_scores, embed_scores, hybrid_scores : dict[str, float], optional
            First-stage retrieval scores.
        labels : list[int], optional
            Ground truth binary click labels (if evaluating).

        Returns
        -------
        ReRankingResult
        """
        cands = [str(cid) for cid in candidate_ids]
        K = len(cands)
        if K == 0:
            return ReRankingResult(
                original_candidate_ids=[],
                reranked_candidate_ids=[],
                reranked_scores=[],
                original_metrics={},
                reranked_metrics={},
            )

        # 1. Extract tabular features
        X = self.feature_pipeline.extract_impression_features(
            user_id=user_id,
            as_of_ts=as_of_ts,
            candidate_ids=cands,
            user_history=user_history,
            user_impressions=user_impressions,
            current_session_id=current_session_id,
            bm25_scores=bm25_scores,
            embed_scores=embed_scores,
            hybrid_scores=hybrid_scores,
        )

        # 2. Predict click probabilities
        pred_scores = self.reranker.predict_scores(X)

        # 3. Stable sort: sort candidates descending by score
        # Using Python's stable sort with enumerate preserves tie order
        scored_pairs = [(pred_scores[i], i, cands[i]) for i in range(K)]
        scored_pairs.sort(key=lambda x: (x[0], -x[1]), reverse=True)

        reranked_ids = [pair[2] for pair in scored_pairs]
        reranked_scores = [float(pair[0]) for pair in scored_pairs]

        # 4. Compute before-vs-after metrics if labels provided
        orig_metrics = {}
        reranked_metrics = {}

        if labels is not None and len(labels) == K:
            # Baseline (Stage 1) pseudo-scores: linearly decreasing ranks
            orig_scores = [float(K - i) for i in range(K)]
            orig_metrics = {
                "AUC": float(auc_score(labels, orig_scores)),
                "MRR": float(mrr(labels, orig_scores)),
                "nDCG@5": float(ndcg_at_k(labels, orig_scores, k=5)),
                "nDCG@10": float(ndcg_at_k(labels, orig_scores, k=10)),
            }

            # Map labels to the reranked candidate order
            label_map = dict(zip(cands, labels))
            reranked_labels = [label_map[cid] for cid in reranked_ids]

            reranked_metrics = {
                "AUC": float(auc_score(reranked_labels, reranked_scores)),
                "MRR": float(mrr(reranked_labels, reranked_scores)),
                "nDCG@5": float(ndcg_at_k(reranked_labels, reranked_scores, k=5)),
                "nDCG@10": float(ndcg_at_k(reranked_labels, reranked_scores, k=10)),
            }

        return ReRankingResult(
            original_candidate_ids=cands,
            reranked_candidate_ids=reranked_ids,
            reranked_scores=reranked_scores,
            original_metrics=orig_metrics,
            reranked_metrics=reranked_metrics,
        )
