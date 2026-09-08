"""Session and within-session behavioural feature extraction for EB-NeRD and MIND.

Implements Assignment 2 Q1 requirements:
1. Session identification:
   - EB-NeRD: uses native `session_id` directly from behavior logs.
   - MIND: reconstructs sessions via a temporal inactivity threshold (default 30 min)
     between consecutive impressions for the same user.
2. Within-session click patterns:
   - `session_click_count`: clicks accumulated within the active session before `as_of_ts`.
   - `session_impression_index`: ordinal index of the impression within the session (0, 1, 2...).
   - `time_since_session_start`: seconds since the session's first event.
   - `time_since_last_impression`: seconds since previous impression in the same session.
3. Dwell time & engagement (dataset-specific):
   - EB-NeRD: extracts `read_time`, `scroll_percentage`, and computes session/historical
     dwell statistics strictly before `as_of_ts`.
   - MIND: dwell time is unavailable in the raw dataset. Flags `dwell_time_available = False`
     with zero/default values (documented known limitation).
4. Position bias modeling:
   - Tracks candidate presentation rank within inview lists.
   - Computes log-discount `1 / log2(2 + rank)` and reciprocal rank `1 / (1 + rank)`.
   - Fits empirical position CTR priors $P(click | position)$ strictly from training data.
5. Strict behavioural-window boundary:
   - All extraction functions enforce that any event with `timestamp >= as_of_ts` is excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import logging
import math
from typing import Any, Sequence

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# Default session inactivity gap for datasets without explicit session_id (e.g. MIND)
DEFAULT_SESSION_TIMEOUT_SECONDS: float = 30 * 60.0  # 30 minutes


# Standard empirical position CTR decay prior (fallback when train split is not yet fit)
# Models the steep presentation bias observed in news recommendation feeds
DEFAULT_POSITION_CTR: list[float] = [
    0.165, 0.112, 0.084, 0.068, 0.055,
    0.046, 0.039, 0.034, 0.030, 0.027,
    0.024, 0.022, 0.020, 0.018, 0.017,
    0.015, 0.014, 0.013, 0.012, 0.011,
]


@dataclass(frozen=True)
class SessionContext:
    """Represents the context of an impression within its session."""
    session_id: str
    impression_index_in_session: int
    clicks_in_session_before_now: int
    dwell_time_in_session_before_now: float  # seconds (EB-NeRD only, 0.0 for MIND)
    mean_scroll_percentage_before_now: float  # 0-100 (EB-NeRD only, 0.0 for MIND)
    time_since_session_start_seconds: float
    time_since_last_impression_seconds: float
    dwell_time_available: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_impression_index": self.impression_index_in_session,
            "session_clicks_so_far": self.clicks_in_session_before_now,
            "session_dwell_time_so_far": self.dwell_time_in_session_before_now,
            "session_mean_scroll_so_far": self.mean_scroll_percentage_before_now,
            "time_since_session_start": self.time_since_session_start_seconds,
            "time_since_last_impression": self.time_since_last_impression_seconds,
            "dwell_time_available": self.dwell_time_available,
        }


@dataclass(frozen=True)
class PositionBiasFeatures:
    """Position bias signals for a candidate at a given display position."""
    position: int
    relative_position: float
    reciprocal_rank: float
    log_position_discount: float
    empirical_position_ctr: float

    def to_dict(self) -> dict[str, float]:
        return {
            "position": float(self.position),
            "relative_position": self.relative_position,
            "reciprocal_rank": self.reciprocal_rank,
            "log_position_discount": self.log_position_discount,
            "empirical_position_ctr": self.empirical_position_ctr,
        }


class PositionBiasModel:
    """Models display position bias and prior click-through rates.

    Candidates shown near the top of the viewport receive systematically higher
    attention and clicks regardless of relevance (presentation bias).
    This class computes position features and fits empirical CTR curves
    strictly from training data.
    """

    def __init__(self, position_ctr_table: dict[int, float] | None = None) -> None:
        if position_ctr_table is not None:
            self._position_ctr = position_ctr_table.copy()
        else:
            self._position_ctr = {i: ctr for i, ctr in enumerate(DEFAULT_POSITION_CTR)}

    @classmethod
    def fit_from_training_behaviors(
        cls,
        behaviors_df: pl.DataFrame,
        max_rank: int = 30,
    ) -> PositionBiasModel:
        """Fit empirical position CTR table strictly from a training behaviors DataFrame.

        Parameters
        ----------
        behaviors_df : pl.DataFrame
            Training split behaviors DataFrame. Must contain `candidates` and `labels`
            as JSON strings or lists, in presentation order.
        max_rank : int
            Maximum presentation position to track.
        """
        position_clicks = np.zeros(max_rank, dtype=np.int64)
        position_inviews = np.zeros(max_rank, dtype=np.int64)

        for row in behaviors_df.iter_rows(named=True):
            raw_cands = row.get("candidates")
            raw_labels = row.get("labels")
            if not raw_cands or not raw_labels:
                continue

            labels = json.loads(raw_labels) if isinstance(raw_labels, str) else list(raw_labels)
            for pos, label in enumerate(labels[:max_rank]):
                position_inviews[pos] += 1
                if label == 1:
                    position_clicks[pos] += 1

        table = {}
        for pos in range(max_rank):
            inviews = position_inviews[pos]
            if inviews > 0:
                # Laplace-smoothed empirical CTR
                ctr = float((position_clicks[pos] + 1) / (inviews + 10))
            else:
                ctr = DEFAULT_POSITION_CTR[pos] if pos < len(DEFAULT_POSITION_CTR) else 0.01
            table[pos] = ctr

        return cls(position_ctr_table=table)

    def get_position_features(
        self,
        position: int,
        total_candidates: int,
    ) -> PositionBiasFeatures:
        """Compute position bias signals for a candidate at index `position`."""
        pos = max(0, position)
        tot = max(1, total_candidates)
        rel_pos = float(pos / max(1, tot - 1))
        recip_rank = 1.0 / (1.0 + pos)
        log_discount = 1.0 / math.log2(2.0 + pos)
        ctr = self._position_ctr.get(
            pos,
            DEFAULT_POSITION_CTR[pos] if pos < len(DEFAULT_POSITION_CTR) else 0.01,
        )

        return PositionBiasFeatures(
            position=pos,
            relative_position=rel_pos,
            reciprocal_rank=recip_rank,
            log_position_discount=log_discount,
            empirical_position_ctr=ctr,
        )


class SessionFeatureExtractor:
    """Extracts session-level and within-session features for user impressions.

    Handles dataset-specific session structures:
    - EB-NeRD: native `session_id`, `read_time`, and `scroll_percentage`.
    - MIND: inactivity-gap sessionization (Delta t <= 30 min), no dwell time.
    """

    def __init__(
        self,
        dataset: str,
        session_timeout_seconds: float = DEFAULT_SESSION_TIMEOUT_SECONDS,
        position_bias_model: PositionBiasModel | None = None,
    ) -> None:
        self.dataset = dataset.lower()
        self.session_timeout_seconds = session_timeout_seconds
        self.position_bias_model = position_bias_model or PositionBiasModel()

    def extract_session_context(
        self,
        user_id: str,
        as_of_ts: datetime,
        user_impressions: Sequence[dict[str, Any]],
        current_session_id: str | int | None = None,
    ) -> SessionContext:
        """Extract within-session context for an impression at `as_of_ts`.

        Parameters
        ----------
        user_id : str
            User identifier.
        as_of_ts : datetime
            Cutoff timestamp. NO event with timestamp >= as_of_ts will be considered!
        user_impressions : list[dict]
            List of all known impressions for this user, with keys:
            `timestamp` (datetime or ISO str), `impression_id`,
            `session_id` (optional), `read_time` (optional),
            `scroll_percentage` (optional), `labels` (optional).
        current_session_id : str or int, optional
            Explicit session_id of current impression if known (EB-NeRD).

        Returns
        -------
        SessionContext
            Within-session statistics strictly preceding `as_of_ts`.
        """
        # 1. Filter out all impressions at or after as_of_ts (strict anti-leakage)
        prior_impressions = []
        for imp in user_impressions:
            ts = imp.get("timestamp")
            if ts is None:
                continue
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            if ts < as_of_ts:
                prior_impressions.append({**imp, "timestamp": ts})

        # Sort chronologically
        prior_impressions.sort(key=lambda x: x["timestamp"])

        if self.dataset == "ebnerd" and current_session_id is not None:
            # EB-NeRD: Filter to impressions sharing the explicit session_id
            session_str = str(current_session_id)
            session_imps = [
                imp for imp in prior_impressions
                if str(imp.get("session_id")) == session_str
            ]
        else:
            # MIND (or EB-NeRD fallback): Reconstruct session via backward inactivity gap
            session_imps = []
            if prior_impressions:
                last_ts = as_of_ts
                session_window = []
                for imp in reversed(prior_impressions):
                    gap = (last_ts - imp["timestamp"]).total_seconds()
                    if gap <= self.session_timeout_seconds:
                        session_window.append(imp)
                        last_ts = imp["timestamp"]
                    else:
                        break
                session_imps = list(reversed(session_window))
                if session_imps:
                    start_iso = session_imps[0]["timestamp"].isoformat()
                    session_str = f"{user_id}_{start_iso}"
                else:
                    session_str = f"{user_id}_{as_of_ts.isoformat()}"
            else:
                session_str = f"{user_id}_{as_of_ts.isoformat()}"

        # 2. Compute within-session metrics
        imp_index = len(session_imps)
        total_clicks = 0
        total_dwell = 0.0
        scroll_percentages = []

        dwell_available = (self.dataset == "ebnerd")

        for imp in session_imps:
            # Clicks
            labels = imp.get("labels")
            if labels is not None:
                if isinstance(labels, str):
                    labels_list = json.loads(labels)
                else:
                    labels_list = list(labels)
                total_clicks += sum(1 for l in labels_list if l == 1)

            # Dwell time & scroll (EB-NeRD)
            if dwell_available:
                rt = imp.get("read_time")
                if rt is not None:
                    try:
                        total_dwell += float(rt)
                    except (ValueError, TypeError):
                        pass
                sp = imp.get("scroll_percentage")
                if sp is not None:
                    try:
                        scroll_percentages.append(float(sp))
                    except (ValueError, TypeError):
                        pass

        mean_scroll = float(np.mean(scroll_percentages)) if scroll_percentages else 0.0

        if session_imps:
            first_ts = session_imps[0]["timestamp"]
            last_ts = session_imps[-1]["timestamp"]
            time_since_start = max(0.0, (as_of_ts - first_ts).total_seconds())
            time_since_last = max(0.0, (as_of_ts - last_ts).total_seconds())
        else:
            time_since_start = 0.0
            time_since_last = -1.0  # Sentinel for first impression in session

        return SessionContext(
            session_id=session_str,
            impression_index_in_session=imp_index,
            clicks_in_session_before_now=total_clicks,
            dwell_time_in_session_before_now=total_dwell if dwell_available else 0.0,
            mean_scroll_percentage_before_now=mean_scroll if dwell_available else 0.0,
            time_since_session_start_seconds=time_since_start,
            time_since_last_impression_seconds=time_since_last,
            dwell_time_available=dwell_available,
        )

    def extract_candidate_position_features(
        self,
        candidate_ids: Sequence[str],
    ) -> list[PositionBiasFeatures]:
        """Extract position bias features for an ordered list of candidates as presented."""
        total = len(candidate_ids)
        return [
            self.position_bias_model.get_position_features(pos, total)
            for pos in range(total)
        ]
