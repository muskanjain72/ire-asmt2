"""Unit tests for the Two-Stage Retrieve-then-Rank re-ranking pipeline."""

from datetime import datetime
import math
from pathlib import Path
import tempfile
import numpy as np
import polars as pl
import pytest

from src.reranking.feature_pipeline import FEATURE_NAMES, ReRankFeaturePipeline
from src.reranking.rerank_pipeline import TwoStageRetrieveThenRank
from src.reranking.train_reranker import GBDTReranker, build_training_dataset


class MockArticleStore:
    def __init__(self, articles: dict[str, dict]):
        self.articles = articles

    def get_article(self, article_id: str):
        return self.articles.get(article_id)

    def get_articles_batch(self, article_ids: list[str], columns=None):
        out = []
        for aid in article_ids:
            if aid in self.articles:
                row = self.articles[aid]
                if columns:
                    out.append({k: row.get(k) for k in columns})
                else:
                    out.append(row.copy())
        return out


@pytest.fixture
def mock_pipeline():
    mock_articles = {
        "A1": {"article_id": "A1", "category": "news", "subcategory": "world", "published_at": "2023-05-20T08:00:00"},
        "A2": {"article_id": "A2", "category": "tech", "subcategory": "ai", "published_at": "2023-05-20T09:00:00"},
        "A3": {"article_id": "A3", "category": "sports", "subcategory": "soccer", "published_at": "2023-05-20T10:00:00"},
    }
    store = MockArticleStore(mock_articles)
    return ReRankFeaturePipeline(dataset="ebnerd", article_store=store)


def test_rerank_feature_pipeline_shape_and_columns(mock_pipeline):
    """Verify feature pipeline outputs correct matrix dimensions and non-null values."""
    as_of = datetime(2023, 5, 20, 12, 0, 0)
    cands = ["A1", "A2", "A3"]
    user_hist = [{"article_id": "A1", "clicked_at": "2023-05-19T12:00:00"}]

    X = mock_pipeline.extract_impression_features(
        user_id="U1",
        as_of_ts=as_of,
        candidate_ids=cands,
        user_history=user_hist,
        bm25_scores={"A1": 5.0, "A2": 2.0},
        embed_scores={"A1": 0.8, "A3": 0.6},
    )

    assert X.shape == (3, len(FEATURE_NAMES))
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()


def test_gbdt_reranker_fit_and_predict():
    """Verify GBDTReranker trains and outputs probability scores."""
    np.random.seed(42)
    n_samples = 200
    n_features = len(FEATURE_NAMES)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Synthetic target: correlate with feature 0 (retrieval_bm25_score) and 24 (category affinity)
    prob = 1.0 / (1.0 + np.exp(-(0.8 * X[:, 0] + 0.6 * X[:, 24])))
    y = (np.random.rand(n_samples) < prob).astype(np.int64)

    # Ensure at least 1 positive and 1 negative
    y[0] = 1
    y[1] = 0

    reranker = GBDTReranker(n_estimators=20, max_depth=3)
    reranker.fit(X, y)

    scores = reranker.predict_scores(X[:10])
    assert len(scores) == 10
    assert (scores >= 0.0).all() and (scores <= 1.0).all()

    importances = reranker.get_feature_importances()
    assert len(importances) == n_features
    assert math.isclose(sum(importances.values()), 1.0, rel_tol=1e-3)


def test_two_stage_retrieve_then_rank_promotes_relevant_candidate(mock_pipeline):
    """Verify that re-ranking promotes high-affinity candidate and improves metrics."""
    as_of = datetime(2023, 5, 20, 12, 0, 0)
    cands = ["A1", "A2", "A3"]  # Stage 1 order
    labels = [0, 0, 1]           # Ground truth: only A3 is clicked

    # Train a simple model that rewards candidate A3 features
    # (A3 is sports/soccer, user loves sports)
    user_hist = [
        {"article_id": "A3", "clicked_at": "2023-05-19T10:00:00"},
        {"article_id": "A3", "clicked_at": "2023-05-20T08:00:00"},
    ]

    # Train model using extracted features so all 28 columns match the feature pipeline distribution
    X_imp = mock_pipeline.extract_impression_features(
        user_id="U_SPORTS",
        as_of_ts=as_of,
        candidate_ids=cands,
        user_history=user_hist,
    )
    X_train = np.vstack([X_imp] * 20)
    y_train = np.array(labels * 20, dtype=np.int64)

    reranker = GBDTReranker(n_estimators=30, max_depth=3)
    reranker.fit(X_train, y_train)

    pipeline = TwoStageRetrieveThenRank(mock_pipeline, reranker)
    result = pipeline.rerank_candidates(
        user_id="U_SPORTS",
        as_of_ts=as_of,
        candidate_ids=cands,
        user_history=user_hist,
        labels=labels,
    )

    # A3 should be promoted to rank 0 (top position)
    assert result.reranked_candidate_ids[0] == "A3"
    assert result.reranked_metrics["MRR"] == 1.0
    assert result.reranked_metrics["nDCG@5"] == 1.0
    # Baseline Stage 1 had A3 at position 2 (0-indexed) -> MRR was 1/3 ≈ 0.333
    assert result.original_metrics["MRR"] < 0.5
    assert result.reranked_metrics["MRR"] > result.original_metrics["MRR"]


def test_stable_tie_breaking(mock_pipeline):
    """Verify that candidates with identical scores preserve original Stage 1 order."""
    class DummyReranker:
        def predict_scores(self, X):
            return np.array([0.5, 0.5, 0.5], dtype=np.float64)

    pipeline = TwoStageRetrieveThenRank(mock_pipeline, DummyReranker())
    result = pipeline.rerank_candidates(
        user_id="U1",
        as_of_ts=datetime(2023, 5, 20, 12, 0, 0),
        candidate_ids=["C1", "C2", "C3"],
    )
    # Tied scores preserve original order
    assert result.reranked_candidate_ids == ["C1", "C2", "C3"]


def test_save_and_load_reranker():
    """Verify model serialization and deserialization via joblib."""
    np.random.seed(42)
    X = np.random.randn(50, len(FEATURE_NAMES)).astype(np.float32)
    y = np.random.randint(0, 2, size=50).astype(np.int64)
    y[0], y[1] = 1, 0

    reranker = GBDTReranker(n_estimators=10, max_depth=2)
    reranker.fit(X, y)
    scores_before = reranker.predict_scores(X[:5])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "reranker.joblib"
        reranker.save(path)
        assert path.exists()

        loaded = GBDTReranker.load(path)
        scores_after = loaded.predict_scores(X[:5])
        np.testing.assert_allclose(scores_before, scores_after)
