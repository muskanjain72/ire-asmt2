"""Unit tests for Q1 behavioural, session, and click-history feature extraction."""

from datetime import datetime, timedelta
import math
import polars as pl
import pytest

from src.feature_store.behavioral_features import (
    BehavioralFeatureExtractor,
    CandidateBehavioralFeatures,
    UserClickHistorySummary,
    compute_recency_weights,
    POSITION_DECAY_FACTOR,
    TIME_DECAY_LAMBDA,
)
from src.feature_store.session_features import (
    PositionBiasFeatures,
    PositionBiasModel,
    SessionContext,
    SessionFeatureExtractor,
)


class MockArticleStore:
    """In-memory article store for testing without disk Parquet files."""
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


def test_recency_weights_time_decay():
    """Verify continuous time-based exponential decay formula."""
    as_of = datetime(2023, 5, 20, 12, 0, 0)
    # 7 days before -> delta = 7*86400 -> exp(-1) ≈ 0.367879
    ts_7d_ago = as_of - timedelta(days=7)
    # 0 days before -> delta = 0 -> exp(0) = 1.0
    ts_now = as_of

    history = [
        {"article_id": "1", "clicked_at": ts_7d_ago.isoformat()},
        {"article_id": "2", "clicked_at": ts_now.isoformat()},
    ]

    weights_unnorm = compute_recency_weights(
        history, as_of, mode="time", normalize=False
    )
    assert len(weights_unnorm) == 2
    assert math.isclose(weights_unnorm[0], math.exp(-1.0), rel_tol=1e-4)
    assert math.isclose(weights_unnorm[1], 1.0, rel_tol=1e-4)

    weights_norm = compute_recency_weights(
        history, as_of, mode="time", normalize=True
    )
    assert math.isclose(sum(weights_norm), 1.0, rel_tol=1e-6)
    assert weights_norm[1] > weights_norm[0]


def test_recency_weights_position_decay():
    """Verify discrete position-based exponential decay formula."""
    as_of = datetime(2023, 5, 20, 12, 0, 0)
    history = [
        {"article_id": "1", "clicked_at": None},
        {"article_id": "2", "clicked_at": None},
        {"article_id": "3", "clicked_at": None},
    ]

    weights_unnorm = compute_recency_weights(
        history, as_of, mode="position", position_decay=0.85, normalize=False
    )
    assert len(weights_unnorm) == 3
    # 3 items: pos from recent = 2, 1, 0 -> 0.85^2, 0.85^1, 0.85^0
    assert math.isclose(weights_unnorm[0], 0.85**2, rel_tol=1e-5)
    assert math.isclose(weights_unnorm[1], 0.85**1, rel_tol=1e-5)
    assert math.isclose(weights_unnorm[2], 1.0, rel_tol=1e-5)


def test_recency_weights_unified_mode():
    """Unified mode automatically selects time decay if timestamps exist, else position."""
    as_of = datetime(2023, 5, 20, 12, 0, 0)

    # With timestamps (EB-NeRD style)
    hist_eb = [
        {"article_id": "1", "clicked_at": (as_of - timedelta(days=1)).isoformat()},
        {"article_id": "2", "clicked_at": as_of.isoformat()},
    ]
    w_eb = compute_recency_weights(hist_eb, as_of, mode="unified", normalize=False)
    assert math.isclose(w_eb[0], math.exp(-1.0 / 7.0), rel_tol=1e-4)

    # Without timestamps (MIND style)
    hist_mind = [
        {"article_id": "1", "clicked_at": None},
        {"article_id": "2", "clicked_at": None},
    ]
    w_mind = compute_recency_weights(hist_mind, as_of, mode="unified", normalize=False)
    assert math.isclose(w_mind[0], 0.85, rel_tol=1e-5)
    assert math.isclose(w_mind[1], 1.0, rel_tol=1e-5)


def test_ebnerd_session_features_and_dwell_time():
    """Verify session features and dwell time calculation on EB-NeRD."""
    extractor = SessionFeatureExtractor(dataset="ebnerd")
    as_of = datetime(2023, 5, 20, 14, 0, 0)

    user_imps = [
        {
            "impression_id": "imp_1",
            "timestamp": datetime(2023, 5, 20, 13, 30, 0),
            "session_id": "sess_42",
            "read_time": 25.0,
            "scroll_percentage": 60.0,
            "labels": [1, 0, 0],
        },
        {
            "impression_id": "imp_2",
            "timestamp": datetime(2023, 5, 20, 13, 45, 0),
            "session_id": "sess_42",
            "read_time": 35.0,
            "scroll_percentage": 80.0,
            "labels": [0, 1],
        },
        # Different session
        {
            "impression_id": "imp_old",
            "timestamp": datetime(2023, 5, 19, 10, 0, 0),
            "session_id": "sess_10",
            "read_time": 10.0,
            "scroll_percentage": 20.0,
            "labels": [1],
        },
    ]

    ctx = extractor.extract_session_context(
        user_id="U100",
        as_of_ts=as_of,
        user_impressions=user_imps,
        current_session_id="sess_42",
    )

    assert ctx.session_id == "sess_42"
    assert ctx.impression_index_in_session == 2
    assert ctx.clicks_in_session_before_now == 2
    assert ctx.dwell_time_in_session_before_now == 60.0  # 25 + 35
    assert ctx.mean_scroll_percentage_before_now == 70.0  # (60 + 80) / 2
    assert ctx.dwell_time_available is True
    assert math.isclose(ctx.time_since_session_start_seconds, 1800.0)  # 13:30 to 14:00
    assert math.isclose(ctx.time_since_last_impression_seconds, 900.0)  # 13:45 to 14:00


def test_mind_session_heuristic_inactivity_gap():
    """Verify session reconstruction on MIND using 30-minute inactivity gap."""
    extractor = SessionFeatureExtractor(dataset="mind", session_timeout_seconds=1800.0)
    as_of = datetime(2019, 11, 14, 10, 0, 0)

    user_imps = [
        # Session 1 (more than 30 min before next)
        {
            "impression_id": "imp_1",
            "timestamp": datetime(2019, 11, 14, 8, 0, 0),
            "labels": [1],
        },
        # Session 2
        {
            "impression_id": "imp_2",
            "timestamp": datetime(2019, 11, 14, 9, 40, 0),
            "labels": [0, 1],
        },
        {
            "impression_id": "imp_3",
            "timestamp": datetime(2019, 11, 14, 9, 50, 0),
            "labels": [1],
        },
    ]

    ctx = extractor.extract_session_context(
        user_id="U200",
        as_of_ts=as_of,
        user_impressions=user_imps,
    )

    # imp_2 and imp_3 form the active session (within 30m of each other and as_of)
    assert ctx.impression_index_in_session == 2
    assert ctx.clicks_in_session_before_now == 2
    assert ctx.dwell_time_available is False
    assert ctx.dwell_time_in_session_before_now == 0.0
    assert math.isclose(ctx.time_since_session_start_seconds, 1200.0)  # 9:40 to 10:00 = 20 min


def test_position_bias_model():
    """Verify position bias features and discount factors."""
    model = PositionBiasModel()
    features = model.get_position_features(position=0, total_candidates=10)
    assert features.position == 0
    assert features.relative_position == 0.0
    assert math.isclose(features.reciprocal_rank, 1.0)
    assert math.isclose(features.log_position_discount, 1.0)  # 1 / log2(2) = 1
    assert features.empirical_position_ctr > 0.1

    feat_pos4 = model.get_position_features(position=4, total_candidates=10)
    assert feat_pos4.position == 4
    assert feat_pos4.reciprocal_rank < features.reciprocal_rank
    assert feat_pos4.empirical_position_ctr < features.empirical_position_ctr


def test_position_bias_fit_from_training_behaviors():
    """Verify empirical position CTR estimation from training behaviors."""
    train_df = pl.DataFrame({
        "candidates": ['["A", "B", "C"]', '["D", "E", "F"]', '["G", "H", "I"]'],
        "labels": ['[1, 0, 0]', '[1, 1, 0]', '[1, 0, 0]'],
    })
    model = PositionBiasModel.fit_from_training_behaviors(train_df, max_rank=3)
    feat0 = model.get_position_features(0, 3)
    feat1 = model.get_position_features(1, 3)
    feat2 = model.get_position_features(2, 3)

    # Position 0 has 3/3 clicks, pos 1 has 1/3, pos 2 has 0/3
    assert feat0.empirical_position_ctr > feat1.empirical_position_ctr
    assert feat1.empirical_position_ctr > feat2.empirical_position_ctr


def test_behavioral_feature_extractor_category_affinity_and_freshness():
    """Verify category affinity distribution, freshness, and train-split popularity."""
    mock_articles = {
        "A1": {
            "article_id": "A1",
            "category": "sports",
            "subcategory": "football",
            "published_at": "2023-05-19T10:00:00",
        },
        "A2": {
            "article_id": "A2",
            "category": "sports",
            "subcategory": "football",
            "published_at": "2023-05-19T12:00:00",
        },
        "A3": {
            "article_id": "A3",
            "category": "finance",
            "subcategory": "stocks",
            "published_at": "2023-05-20T08:00:00",
        },
        "CAND_SPORTS": {
            "article_id": "CAND_SPORTS",
            "category": "sports",
            "subcategory": "football",
            "published_at": "2023-05-20T09:00:00",
        },
        "CAND_NEWS": {
            "article_id": "CAND_NEWS",
            "category": "politics",
            "subcategory": "election",
            "published_at": "2023-05-20T10:00:00",
        },
    }
    store = MockArticleStore(mock_articles)
    extractor = BehavioralFeatureExtractor(
        dataset="ebnerd",
        article_store=store,
        train_popularity={"CAND_SPORTS": 50, "CAND_NEWS": 5},
        train_inviews={"CAND_SPORTS": 500, "CAND_NEWS": 100},
    )

    as_of = datetime(2023, 5, 20, 12, 0, 0)
    raw_history = [
        {"article_id": "A1", "clicked_at": "2023-05-19T15:00:00"},
        {"article_id": "A2", "clicked_at": "2023-05-20T08:30:00"},
        {"article_id": "A3", "clicked_at": "2023-05-20T09:00:00"},
    ]

    summary = extractor.summarize_user_history("U1", as_of, raw_history=raw_history)
    assert summary.active_history_len == 3
    # sports has 2 clicks, finance has 1
    assert "sports" in summary.category_distribution
    assert "finance" in summary.category_distribution
    assert summary.category_distribution["sports"] > summary.category_distribution["finance"]
    assert summary.top_category == "sports"

    cand_feats = extractor.extract_candidate_features(
        ["CAND_SPORTS", "CAND_NEWS"],
        summary,
        as_of,
    )
    assert len(cand_feats) == 2

    # CAND_SPORTS
    f_sports = cand_feats[0]
    assert f_sports.article_id == "CAND_SPORTS"
    assert f_sports.category_affinity > 0.5
    assert f_sports.is_top_category_match is True
    assert f_sports.is_top_subcategory_match is True
    assert math.isclose(f_sports.freshness_hours, 3.0)  # 09:00 to 12:00 = 3h
    assert f_sports.freshness_available is True
    assert f_sports.train_popularity_log_clicks == math.log1p(50)

    # CAND_NEWS
    f_news = cand_feats[1]
    assert f_news.article_id == "CAND_NEWS"
    assert f_news.category_affinity == 0.0  # never read politics
    assert f_news.is_top_category_match is False
    assert math.isclose(f_news.freshness_hours, 2.0)  # 10:00 to 12:00 = 2h
