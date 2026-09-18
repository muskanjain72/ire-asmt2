"""Thin orchestration layer for large Codabench result generation.

The heavy lifting deliberately lives in ingestion/parsing/feature-store/retrieval:
  * large-test streaming readers: src.parsing.submission_readers
  * EB-NeRD compact history index: src.feature_store.history_store
  * article embeddings: src.retrieval.embeddings
  * user vectors: src.retrieval.user_representation
  * candidate ranking: src.retrieval.candidate_gen
  * output/schema validation: src.submission.writers
"""

from __future__ import annotations

import argparse
import logging
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.feature_store.history_store import MemoryMappedHistoryStore
from src.parsing.submission_readers import (
    Impression,
    find_ebnerd_test_files,
    find_mind_test_behaviors,
    iter_ebnerd_test,
    iter_mind_test,
)
from src.retrieval.ann import ArticleIndex
from src.retrieval.candidate_gen import rank_candidate_batch
from src.retrieval.embeddings import load_embeddings
from src.retrieval.user_representation import build_mean_user_vectors
from src.submission.package_submission import package_prediction
from src.submission.writers import write_ranked_impression

# Stage-2 GBDT reranker (EB-NeRD only; MIND skips reranking — see Proposal 3 rationale)
try:
    from src.feature_store.article_store import ArticleFeatureStore
    from src.feature_store.behavioral_features import BehavioralFeatureExtractor
    from src.feature_store.session_features import PositionBiasModel, SessionFeatureExtractor
    from src.reranking.feature_pipeline import ReRankFeaturePipeline
    from src.reranking.train_reranker import GBDTReranker
    from src.reranking.rerank_pipeline import TwoStageRetrieveThenRank
    from src.evaluation.eval_reranker import load_dataset_popularity
    HAS_RERANKER = True
except ImportError:
    HAS_RERANKER = False

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _stable_rank_positions(scores):
    """Return 1-based ranks, preserving input order for exact score ties.

    Kept here as a backwards-compatible helper for the submission tests and
    older callers. The actual ranking primitive lives in src.retrieval.
    """
    import numpy as np
    scores_arr = np.asarray(scores, dtype=np.float32)
    order = np.argsort(-scores_arr, kind="stable")
    ranks = np.empty(len(scores_arr), dtype=np.int32)
    ranks[order] = np.arange(1, len(scores_arr) + 1, dtype=np.int32)
    return ranks.tolist()


def _parse_mind_row(line: str) -> Impression:
    """Parse one MIND behavior row for backwards-compatible callers."""
    from src.parsing.submission_readers import _parse_mind_candidates, _parse_mind_history, _parse_mind_time

    parts = line.rstrip("\n").split("\t")
    if len(parts) != 5:
        raise ValueError(f"Expected 5 MIND behavior fields, got {len(parts)}")
    impression_id, user_id, ts_text, history_text, impressions_text = parts
    timestamp = _parse_mind_time(ts_text)
    history = _parse_mind_history(history_text)
    candidates = _parse_mind_candidates(impressions_text)
    if not candidates:
        raise ValueError(f"Impression {impression_id} contains no candidates")
    return Impression(
        impression_id=str(impression_id),
        user_id=str(user_id),
        timestamp=timestamp,
        history=history,
        candidates=candidates,
    )


def validate_prediction_file(path: Path, expected_filename: str | None = None) -> int:
    """Compatibility wrapper around the canonical prediction validator."""
    if expected_filename is not None and path.name != expected_filename:
        raise ValueError(f"Expected filename {expected_filename}, got {path.name}")
    from src.submission.writers import validate_prediction_file as _validate
    return _validate(path)


def validate_zip(path: Path, expected_filename: str) -> None:
    """Validate a submission ZIP contains exactly one valid prediction file."""
    import zipfile
    import io
    if not path.exists():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        if names != [expected_filename]:
            raise ValueError(
                f"Expected ZIP to contain exactly [{expected_filename!r}], got {names!r}"
            )
        raw = zf.read(expected_filename)
        temp = io.StringIO(raw.decode("utf-8"))
        for line_no, line in enumerate(temp, 1):
            if not line.strip():
                raise ValueError(f"Blank prediction line at {line_no}")
            parts = line.rstrip("\n").split(" [", 1)
            if len(parts) != 2 or not parts[1].endswith("]"):
                raise ValueError(f"Malformed prediction line {line_no}")
            ranks = [int(x) for x in parts[1][:-1].split(",")] if parts[1][:-1] else []
            if sorted(ranks) != list(range(1, len(ranks) + 1)):
                raise ValueError(f"Ranks at line {line_no} are not 1..N")


def _find_mind_catalog(raw_dir: Path) -> tuple[list[str], list[str]]:
    """Build the MIND article catalog from train/dev/test news.tsv files.

    Only the fields needed for semantic retrieval are loaded:
    news_id, title, and abstract.
    """
    paths = [
        raw_dir / "MINDlarge_train" / "news.tsv",
        raw_dir / "MINDlarge_dev" / "news.tsv",
        raw_dir / "MINDlarge_test" / "news.tsv",
    ]

    paths = [p for p in paths if p.exists()]

    if not paths:
        paths = sorted(raw_dir.rglob("news.tsv"))

    if not paths:
        raise FileNotFoundError(
            f"No MIND news.tsv found below {raw_dir}"
        )

    rows: dict[str, str] = {}

    for path in paths:
        df = pl.read_csv(
            path,
            separator="\t",
            has_header=False,
            new_columns=[
                "news_id",
                "title",
                "abstract",
            ],
            infer_schema_length=0,
            truncate_ragged_lines=True,
            quote_char=None,
            columns=[0, 3, 4],
        )

        for article_id, title, abstract in df.iter_rows():
            text = str(title or "")
            if abstract:
                text += " " + str(abstract)

            rows[str(article_id)] = text

    article_ids = list(rows.keys())
    article_texts = [rows[aid] for aid in article_ids]

    logger.info(
        "MIND submission catalog: %d unique articles",
        len(article_ids),
    )

    return article_ids, article_texts


def _load_index(dataset: str, args: argparse.Namespace) -> ArticleIndex:
    if dataset == "mind":
        article_ids, texts = _find_mind_catalog(_PROJECT_ROOT / "data" / "raw" / "mind")
        embeddings, id_to_row, _ = load_embeddings(
            "mind",
            "minilm",
            article_ids=article_ids,
            article_texts=texts,
            cache_tag="large",
            batch_size=args.embedding_batch_size,
            device=args.device,
        )
    else:
        embeddings, id_to_row, coverage = load_embeddings(
            "ebnerd",
            args.ebnerd_model,
            article_ids=None,
            cache_tag="large",
        )
        logger.info("EB-NeRD embedding coverage: %.2f%%", coverage * 100)

    article_ids_ordered = [""] * len(id_to_row)
    for aid, idx in id_to_row.items():
        article_ids_ordered[int(idx)] = aid
    return ArticleIndex(embeddings, article_ids_ordered, build_full_index=False)


def _history_batches(
    dataset: str,
    batch: list[Impression],
    eb_history: MemoryMappedHistoryStore | None,
) -> list[list[dict[str, str | None]]]:
    if dataset == "mind":
        return [item.history for item in batch]
    if eb_history is None:
        raise RuntimeError("EB-NeRD history store is missing")
    return eb_history.get_histories([item.user_id for item in batch], [item.timestamp for item in batch])


def _load_reranker_pipeline(dataset: str) -> TwoStageRetrieveThenRank | None:
    """Load the trained GBDT reranker pipeline for Stage-2 re-ranking.

    Returns None if the reranker module is unavailable or the saved model
    doesn't exist, so that submission can fall back to Stage-1 ranking.
    For MIND: always returns None (GBDT hurts MIND — see implementation plan).
    For EB-NeRD: loads the trained ebnerd_reranker.joblib and wires it into
    a feature pipeline backed by the small-scale processed article store.
    """
    if dataset == "mind":
        # MIND GBDT is intentionally skipped: standalone GBDT AUC 0.4823 vs
        # NRMS-only 0.6338. Until retrieval scores are plumbed (Proposal 2),
        # Stage-1 embedding ranking is the best MIND model.
        logger.info("MIND: using Stage-1 embedding ranking (GBDT skipped per Proposal 3).")
        return None

    if not HAS_RERANKER:
        logger.warning("Reranker module unavailable — falling back to Stage-1 ranking.")
        return None

    model_path = _PROJECT_ROOT / "models" / f"{dataset}_reranker.joblib"
    if not model_path.exists():
        logger.warning("No trained reranker at %s — falling back to Stage-1 ranking.", model_path)
        return None

    # Feature stores: use small-scale processed data (article_features.parquet)
    # which holds freshness, category, popularity priors fitted on training split.
    p_dir = _PROJECT_ROOT / "data" / "processed" / dataset
    i_dir = _PROJECT_ROOT / "data" / "interim" / dataset
    art_path = p_dir / "article_features.parquet"
    beh_path = i_dir / "behaviors.parquet"

    if not art_path.exists():
        logger.warning("Article features missing at %s — falling back to Stage-1.", art_path)
        return None

    article_store = ArticleFeatureStore(dataset=dataset, processed_dir=p_dir)

    # Fit position bias and popularity priors from training split (no leakage)
    if beh_path.exists():
        df_beh = pl.read_parquet(beh_path)
        train_df = df_beh.filter(pl.col("split") == "train") if "split" in df_beh.columns else df_beh
        train_clicks, train_inviews = load_dataset_popularity(train_df)
        pos_model = PositionBiasModel.fit_from_training_behaviors(train_df)
    else:
        logger.warning("Behaviors parquet missing at %s — using default position bias priors.", beh_path)
        train_clicks, train_inviews = {}, {}
        pos_model = PositionBiasModel()

    behavioral_extractor = BehavioralFeatureExtractor(
        dataset=dataset,
        article_store=article_store,
        train_popularity=train_clicks,
        train_inviews=train_inviews,
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

    reranker = GBDTReranker.load(model_path)
    logger.info("Loaded GBDT reranker from %s (%d features)", model_path, len(reranker.feature_names))
    return TwoStageRetrieveThenRank(feature_pipeline=feature_pipeline, reranker=reranker)


def _process_batch(
    batch: list[Impression],
    dataset: str,
    index: ArticleIndex,
    eb_history: MemoryMappedHistoryStore | None,
    history_cap: int,
    handle,
    reranker_pipeline: TwoStageRetrieveThenRank | None = None,
) -> None:
    """Rank one batch of impressions and write predictions.

    Stage-1: cosine-similarity ranking over the article embedding index.
    Stage-2 (EB-NeRD only, when reranker_pipeline is set): GBDT re-ranking
    using the Stage-1 embed scores + behavioral/article/position features.
    """
    histories = _history_batches(dataset, batch, eb_history)
    candidates = [item.candidates for item in batch]
    vectors = build_mean_user_vectors(histories, index, history_cap=history_cap)

    if reranker_pipeline is None:
        # Stage-1 only: pure embedding similarity ranking
        ranked = rank_candidate_batch(vectors, candidates, index)
        for item, ordered in zip(batch, ranked):
            write_ranked_impression(handle, item.impression_id, item.candidates, ordered)
    else:
        # Stage-1 + Stage-2: use embed scores from Stage-1 as input to the GBDT
        from src.retrieval.ann import ArticleIndex as _AI
        for item, user_vector, hist_list in zip(batch, vectors, histories):
            cands = item.candidates
            if not cands:
                write_ranked_impression(handle, item.impression_id, [], [])
                continue

            # Stage-1: score all candidates by embedding similarity
            stage1_results = index.search_restricted(user_vector, cands, k=len(cands))
            embed_scores: dict[str, float] = {aid: float(score) for aid, score in stage1_results}

            # Stage-2: GBDT reranker using embed_scores as retrieval_embed_sim feature
            # History entries are dicts with at least "article_id"; no timestamps in test set.
            user_history = [
                {"article_id": str(h["article_id"]), "clicked_at": None}
                for h in hist_list
                if h.get("article_id")
            ]
            as_of = item.timestamp or datetime(2023, 5, 22, 12, 0, 0)

            result = reranker_pipeline.rerank_candidates(
                user_id=item.user_id,
                as_of_ts=as_of,
                candidate_ids=cands,
                user_history=user_history,
                user_impressions=[],   # test set has no prior session context
                current_session_id=None,
                embed_scores=embed_scores,
                labels=None,
            )
            write_ranked_impression(
                handle, item.impression_id, item.candidates, result.reranked_candidate_ids
            )


def generate_submission(dataset: str, args: argparse.Namespace) -> tuple[Path, Path, int, float]:
    started = time.time()
    index = _load_index(dataset, args)
    logger.info("Loaded submission index: %s", index)

    output_dir = _PROJECT_ROOT / "submissions" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_filename = (
        "predictions.txt"
        if dataset == "ebnerd"
        else "prediction.txt"
    )
    prediction_path = output_dir / prediction_filename
    zip_path = output_dir / f"{dataset}_submission.zip"

    # Load Stage-2 GBDT reranker pipeline (None for MIND, or if --no-rerank)
    no_rerank = getattr(args, "no_rerank", False)
    reranker_pipeline = None if no_rerank else _load_reranker_pipeline(dataset)
    if reranker_pipeline is not None:
        logger.info("Stage-2 GBDT reranker active for %s.", dataset)
    else:
        logger.info("Stage-1 only (no Stage-2 reranker) for %s.", dataset)

    eb_history = None
    if dataset == "ebnerd":
        _, history_path = find_ebnerd_test_files(_PROJECT_ROOT / "data" / "raw" / "ebnerd")
        history_dir = _PROJECT_ROOT / "data" / "processed" / "submission" / "ebnerd_history"
        eb_history = MemoryMappedHistoryStore.build(history_path, history_dir, force=args.rebuild_history_index)
        batches = iter_ebnerd_test(
            find_ebnerd_test_files(_PROJECT_ROOT / "data" / "raw" / "ebnerd")[0],
            batch_size=args.batch_size,
        )
    else:
        test_path = find_mind_test_behaviors(_PROJECT_ROOT / "data" / "raw" / "mind")
        batches = iter_mind_test(test_path, batch_size=args.batch_size)

    row_count = 0
    with prediction_path.open("w", encoding="utf-8") as handle:
        for batch in batches:
            _process_batch(
                batch, dataset, index, eb_history, args.history_cap, handle,
                reranker_pipeline=reranker_pipeline,
            )
            row_count += len(batch)
            if row_count % max(args.batch_size * 5, 100_000) < len(batch):
                logger.info("Generated %d predictions", row_count)

    package_prediction(prediction_path, zip_path)
    elapsed = time.time() - started
    logger.info("Generated %d predictions in %.1fs", row_count, elapsed)
    return prediction_path, zip_path, row_count, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Codabench result ZIP")
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--history-cap", type=int, default=20)
    parser.add_argument("--ebnerd-model", choices=["w2v", "bert"], default="w2v")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--device", default=None, help="sentence-transformers device, e.g. cuda or cpu")
    parser.add_argument("--rebuild-history-index", action="store_true")
    parser.add_argument(
        "--no-rerank", action="store_true",
        help="Skip Stage-2 GBDT reranking (for ablation/debugging — produces Stage-1 only output)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    datasets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        prediction, zip_path, rows, elapsed = generate_submission(dataset, args)
        print(f"\n{dataset.upper()}: {rows:,} rows, {elapsed/60:.1f} min")
        print(f"  prediction: {prediction}")
        print(f"  submission: {zip_path}")


if __name__ == "__main__":
    main()
