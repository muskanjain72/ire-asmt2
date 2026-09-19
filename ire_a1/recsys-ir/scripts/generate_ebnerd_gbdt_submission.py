"""Generate official EB-NeRD Codabench submission using verified GBDT canonical pipeline.

Model checkpoint: models/ebnerd_reranker.joblib (canonical validation AUC 0.6098).
Large test set: data/raw/ebnerd/ebnerd_testset/test/behaviors.parquet (13,536,710 impressions).
Output file: submissions/ebnerd/predictions.txt
Zipped file: submissions/ebnerd/ebnerd_submission.zip
"""

from __future__ import annotations

import argparse
from datetime import datetime
import gc
import json
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np
import polars as pl
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import interim_dir, processed_dir
from src.evaluation.eval_reranker import load_dataset_popularity
from src.feature_store.article_store import ArticleFeatureStore
from src.feature_store.behavioral_features import BehavioralFeatureExtractor
from src.feature_store.session_features import PositionBiasModel, SessionFeatureExtractor
from src.reranking.feature_pipeline import FEATURE_NAMES, ReRankFeaturePipeline
from src.reranking.train_reranker import GBDTReranker
from src.submission.package_submission import package_prediction
from src.submission.writers import validate_prediction_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("ebnerd_gbdt_submission")

EXPECTED_EBNERD_ROWS = 13_536_710
EXPECTED_CHECKPOINT_SIZE = 325_422


def verify_prerequisites() -> tuple[Path, Path]:
    eb_dir = PROJECT_ROOT / "data" / "raw" / "ebnerd"
    test_zip = eb_dir / "ebnerd_testset.zip"
    test_behaviors = eb_dir / "ebnerd_testset" / "test" / "behaviors.parquet"
    checkpoint = PROJECT_ROOT / "models" / "ebnerd_reranker.joblib"

    if not test_zip.exists():
        raise FileNotFoundError(f"CRITICAL: EB-NeRD large test archive missing at {test_zip}")
    if not test_behaviors.exists():
        raise FileNotFoundError(f"CRITICAL: EB-NeRD test behaviors missing at {test_behaviors}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"CRITICAL: Canonical GBDT checkpoint missing at {checkpoint}")

    ckpt_size = checkpoint.stat().st_size
    logger.info("Confirmed ebnerd_testset.zip present: %d bytes", test_zip.stat().st_size)
    logger.info("Confirmed checkpoint present: %s (%d bytes)", checkpoint, ckpt_size)

    return test_behaviors, checkpoint


def setup_pipeline(checkpoint_path: Path) -> tuple[ReRankFeaturePipeline, GBDTReranker]:
    dataset = "ebnerd"
    scale = "small"
    p_dir = processed_dir(dataset, scale)
    i_dir = interim_dir(dataset, scale)
    beh_path = i_dir / "behaviors.parquet"
    art_path = p_dir / "article_features.parquet"

    logger.info("Loading canonical GBDT checkpoint from: %s", checkpoint_path)
    reranker = GBDTReranker.load(checkpoint_path)
    logger.info("Loaded GBDT model (%d features)", len(reranker.feature_names))

    logger.info("Setting up feature extractors and in-memory article store...")
    article_store = ArticleFeatureStore(dataset, processed_dir=p_dir, scale=scale)

    # Pre-populate article cache from article_features.parquet to guarantee O(1) in-RAM lookup
    df_art = pl.read_parquet(art_path)
    for r in df_art.iter_rows(named=True):
        article_store._article_cache[str(r["article_id"])] = r
    logger.info("Pre-warmed article store cache with %d articles.", len(article_store._article_cache))

    # Priors strictly from training split (no leakage)
    df_beh = pl.read_parquet(beh_path)
    train_df = df_beh.filter(pl.col("split") == "train") if "split" in df_beh.columns else df_beh
    train_clicks, train_inviews = load_dataset_popularity(train_df)
    pos_model = PositionBiasModel.fit_from_training_behaviors(train_df)
    logger.info("Fit position bias model and popularity priors on training split (%d clicks, %d inviews).",
                len(train_clicks), len(train_inviews))

    behavioral_extractor = BehavioralFeatureExtractor(
        dataset=dataset,
        article_store=article_store,
        train_popularity=train_clicks,
        train_inviews=train_inviews,
        strict_time=False,
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

    return feature_pipeline, reranker


def generate_ebnerd_predictions(
    test_behaviors_path: Path,
    feature_pipeline: ReRankFeaturePipeline,
    reranker: GBDTReranker,
    output_path: Path,
    chunk_size: int = 50_000,
) -> int:
    logger.info("Streaming and re-ranking %d impressions to %s (chunk_size=%d)...",
                EXPECTED_EBNERD_ROWS, output_path, chunk_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    article_cache = feature_pipeline.article_store._article_cache
    pf = pq.ParquetFile(test_behaviors_path)
    columns = ["impression_id", "user_id", "impression_time", "article_ids_inview"]

    started_time = time.time()
    processed_count = 0
    default_ts = datetime(2023, 5, 22, 12, 0, 0)

    with output_path.open("w", encoding="utf-8", buffering=1024 * 1024 * 16) as out_f:
        for batch_record in pf.iter_batches(batch_size=chunk_size, columns=columns):
            pydict = batch_record.to_pydict()
            batch_imp_ids = pydict["impression_id"]
            batch_uids = pydict["user_id"]
            batch_ts = pydict["impression_time"]
            batch_cands = pydict["article_ids_inview"]
            batch_len = len(batch_imp_ids)

            X_list = []
            cand_counts = []
            valid_batch_indices = []

            for i in range(batch_len):
                cands_raw = batch_cands[i]
                if not cands_raw:
                    cand_counts.append(0)
                    continue

                cands = [str(c) for c in cands_raw]
                cand_counts.append(len(cands))
                valid_batch_indices.append(i)

                # Ensure missing candidates are cached as empty dict to avoid DuckDB disk calls
                for c in cands:
                    if c not in article_cache:
                        article_cache[c] = {"article_id": c}

                as_of = batch_ts[i] if batch_ts[i] is not None else default_ts
                uid_str = str(batch_uids[i])

                X_imp = feature_pipeline.extract_impression_features(
                    user_id=uid_str,
                    as_of_ts=as_of,
                    candidate_ids=cands,
                    user_history=[],
                    user_impressions=[],
                    current_session_id=None,
                )
                X_list.append(X_imp)

            if X_list:
                X_mat = np.vstack(X_list)
                # Batch predict probabilities
                batch_scores = reranker.model.predict_proba(X_mat)[:, 1]
            else:
                batch_scores = np.empty((0,), dtype=np.float32)

            # Output lines
            lines = []
            score_offset = 0
            for i in range(batch_len):
                k = cand_counts[i]
                imp_id = batch_imp_ids[i]
                if k == 0:
                    lines.append(f"{imp_id} []\n")
                    continue

                scores = batch_scores[score_offset : score_offset + k]
                score_offset += k

                # 1-based ranks matching Codabench: rank 1 is highest score
                order = np.argsort(-scores, kind="stable")
                ranks = np.empty(k, dtype=np.int32)
                ranks[order] = np.arange(1, k + 1, dtype=np.int32)

                ranks_str = ",".join(map(str, ranks))
                lines.append(f"{imp_id} [{ranks_str}]\n")

            out_f.writelines(lines)
            processed_count += batch_len

            if processed_count % (chunk_size * 10) < chunk_size or processed_count == EXPECTED_EBNERD_ROWS:
                elapsed = time.time() - started_time
                speed = processed_count / max(1.0, elapsed)
                eta_min = (EXPECTED_EBNERD_ROWS - processed_count) / max(1.0, speed) / 60.0
                pct = processed_count / EXPECTED_EBNERD_ROWS * 100
                logger.info(
                    "  [%d / %d] (%.1f%%) | Speed: %.1f impr/s | Elapsed: %.1fm | ETA: %.1fm",
                    processed_count,
                    EXPECTED_EBNERD_ROWS,
                    pct,
                    speed,
                    elapsed / 60.0,
                    eta_min,
                )

    total_elapsed = time.time() - started_time
    logger.info(
        "Scored all %d EB-NeRD impressions in %.2fs (%.1f minutes, %.1f impr/s)",
        processed_count,
        total_elapsed,
        total_elapsed / 60.0,
        processed_count / total_elapsed,
    )
    return processed_count


def main():
    test_behaviors, checkpoint = verify_prerequisites()

    feature_pipeline, reranker = setup_pipeline(checkpoint)

    output_dir = PROJECT_ROOT / "submissions" / "ebnerd"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.txt"
    zip_path = output_dir / "ebnerd_submission.zip"

    actual_rows = generate_ebnerd_predictions(
        test_behaviors, feature_pipeline, reranker, prediction_path, chunk_size=50_000
    )

    # Strict row count gate
    logger.info("Verifying row count: actual=%d vs expected=%d", actual_rows, EXPECTED_EBNERD_ROWS)
    if actual_rows != EXPECTED_EBNERD_ROWS:
        raise ValueError(
            f"CRITICAL ROW COUNT MISMATCH: {actual_rows} != {EXPECTED_EBNERD_ROWS}! Aborting packaging!"
        )

    # Validate output format
    logger.info("Running validate_prediction_file on %s...", prediction_path)
    validated_rows = validate_prediction_file(prediction_path, expected_rows=EXPECTED_EBNERD_ROWS)
    logger.info("Prediction file validation PASSED with %d rows.", validated_rows)

    # Package into ZIP
    logger.info("Packaging %s into %s...", prediction_path, zip_path)
    package_prediction(prediction_path, zip_path)
    logger.info("Submission ZIP created successfully: %s (%d bytes)", zip_path, zip_path.stat().st_size)


if __name__ == "__main__":
    main()
