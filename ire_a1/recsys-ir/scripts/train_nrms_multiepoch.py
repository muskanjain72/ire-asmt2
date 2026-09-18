"""Multi-Epoch NRMS Training & Progression Evaluation Pipeline.

Trains the official EB-NeRD NRMS neural baseline across multiple epochs (e.g. 1 to 5),
logging training loss and evaluating validation ranking metrics (AUC, MRR, nDCG@5, nDCG@10)
at the conclusion of each epoch. Checkpoints weights per epoch and trains the Stage 2 GBDT
re-ranker on the best epoch's representations.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import gc
import json
import logging
from pathlib import Path
import sys
import time

# RecSys-IR codebase root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

import numpy as np
import polars as pl
import tensorflow as tf

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Configured GPU memory growth on %s", gpu.name)
        except RuntimeError as e:
            logger.warning("GPU configuration error: %s", e)
else:
    logger.warning("No GPU detected by TensorFlow, falling back to CPU.")

from ebrec.models.newsrec import NRMSModel
from ebrec.models.newsrec.dataloader import NRMSDataLoader, NRMSDataLoaderPretransform
from ebrec.models.newsrec.model_config import hparams_nrms, hparams_to_dict
from ebrec.utils._constants import (
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
)

# Import helper functions from train_official_ebnerd_nrms
from scripts.train_official_ebnerd_nrms import (
    prepare_articles_and_embeddings,
    prepare_training_behaviors,
    evaluate_nrms_baseline,
    train_and_eval_gbdt_reranker,
    run_statistical_significance_tests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-epoch EB-NeRD NRMS training pipeline.")
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "ebnerd" / "ebnerd_demo"),
        help="Path to EB-NeRD dataset directory containing articles.parquet, train, validation.",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "ebnerd"),
        help="Directory to read/store processed article features.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(PROJECT_ROOT / "models" / "nrms_multiepoch"),
        help="Directory to save per-epoch NRMS weights and models.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(PROJECT_ROOT / "results"),
        help="Directory to save evaluation progression CSV summaries.",
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=50000,
        help="Number of informative training impressions (0 for full train split).",
    )
    parser.add_argument(
        "--val_samples",
        type=int,
        default=500,
        help="Number of validation impressions to evaluate ranking metrics per epoch.",
    )
    parser.add_argument(
        "--gbdt_train_samples",
        type=int,
        default=1000,
        help="Number of training impressions to score and train GBDT re-ranker on.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Total number of epochs to train.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for NRMS training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def run_multiepoch_pipeline(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path)
    processed_dir = Path(args.processed_dir)
    model_dir = Path(args.model_dir)
    results_dir = Path(args.results_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    logger.info("Starting Multi-Epoch NRMS Pipeline (Target Epochs: %d)", args.epochs)

    # 1. Preprocess articles & embeddings
    article_mapping, compact_embeddings, df_articles = prepare_articles_and_embeddings(
        data_path=data_path,
        processed_dir=processed_dir,
        max_title_length=30,
    )

    # 2. Prepare training behaviors
    sample_size = None if args.train_samples == 0 else args.train_samples
    df_train_raw, df_train_npr = prepare_training_behaviors(
        train_path=data_path / "train",
        history_size=20,
        sample_size=sample_size,
        npratio=4,
        seed=args.seed,
    )

    # 3. Setup DataLoader
    n_full = (len(df_train_npr) // args.batch_size) * args.batch_size
    if n_full < len(df_train_npr):
        df_train_npr = df_train_npr.slice(0, n_full)

    train_loader = NRMSDataLoaderPretransform(
        behaviors=df_train_npr,
        article_dict=article_mapping,
        unknown_representation="zeros",
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        eval_mode=False,
        batch_size=args.batch_size,
    )

    # 4. Instantiate NRMS Model
    hparams_nrms.title_size = 30
    hparams_nrms.history_size = 20
    hparams_nrms.head_num = 16
    hparams_dim = 16
    hparams_nrms.head_dim = hparams_dim
    hparams_nrms.attention_hidden_dim = 200
    hparams_nrms.loss = "cross_entropy_loss"
    hparams_nrms.dropout = 0.2
    hparams_nrms.learning_rate = args.learning_rate

    nrms = NRMSModel(
        hparams=hparams_nrms,
        word2vec_embedding=compact_embeddings,
        seed=args.seed,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    nrms.model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["AUC"],
    )

    # 5. Training loop across epochs with per-epoch eval
    progression_records = []
    best_auc = 0.0
    best_epoch = 1

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        logger.info("-" * 60)
        logger.info("EPOCH %d / %d", epoch, args.epochs)
        logger.info("-" * 60)

        history = nrms.model.fit(
            train_loader,
            epochs=1,
            verbose=1,
        )
        train_loss = float(history.history["loss"][0])
        train_auc = float(history.history["auc"][0]) if "auc" in history.history else 0.0
        epoch_dur = time.time() - epoch_start

        # Save checkpoint weights for this epoch
        epoch_weights_path = model_dir / f"nrms_weights_epoch_{epoch}.weights.h5"
        nrms.model.save_weights(epoch_weights_path)
        logger.info("Saved epoch %d weights to %s", epoch, epoch_weights_path)

        # Evaluate on validation split
        df_val_eval, metrics_dict, summary_means = evaluate_nrms_baseline(
            nrms=nrms,
            val_path=data_path / "validation",
            article_mapping=article_mapping,
            val_samples=args.val_samples,
            eval_batch_size=args.eval_batch_size,
            history_size=20,
            seed=args.seed,
        )

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_auc": round(train_auc, 4),
            "val_auc": round(summary_means["AUC"], 4),
            "val_mrr": round(summary_means["MRR"], 4),
            "val_ndcg5": round(summary_means["nDCG@5"], 4),
            "val_ndcg10": round(summary_means["nDCG@10"], 4),
            "epoch_duration_sec": round(epoch_dur, 1),
        }
        progression_records.append(record)
        logger.info(
            "Epoch %d Summary: Train Loss=%.4f | Val AUC=%.4f | Val MRR=%.4f | Duration=%.1fs",
            epoch, train_loss, summary_means["AUC"], summary_means["MRR"], epoch_dur,
        )

        if summary_means["AUC"] > best_auc:
            best_auc = summary_means["AUC"]
            best_epoch = epoch
            # Save as canonical best
            best_weights_path = model_dir / "nrms_weights_best.weights.h5"
            nrms.model.save_weights(best_weights_path)

    # Save progression to CSV
    df_progression = pl.DataFrame(progression_records)
    prog_csv = results_dir / "nrms_epochs_progression.csv"
    df_progression.write_csv(prog_csv)
    logger.info("Saved multi-epoch training progression to %s", prog_csv)
    print("\n" + "=" * 60)
    print(f"MULTI-EPOCH NRMS TRAINING PROGRESSION (Best Epoch: {best_epoch} with Val AUC={best_auc:.4f}):")
    print("=" * 60)
    print(df_progression)

    # 6. Train GBDT re-ranker on the best model
    logger.info("Training GBDT re-ranker using representations from Best Epoch (%d)...", best_epoch)
    best_weights = model_dir / f"nrms_weights_epoch_{best_epoch}.weights.h5"
    nrms.model.load_weights(best_weights)

    # Re-evaluate validation set to ensure exact alignments
    df_val_best, metrics_nrms, summary_nrms = evaluate_nrms_baseline(
        nrms=nrms,
        val_path=data_path / "validation",
        article_mapping=article_mapping,
        val_samples=args.val_samples,
        eval_batch_size=args.eval_batch_size,
        history_size=20,
        seed=args.seed,
    )

    reranker, metrics_gbdt, summary_gbdt = train_and_eval_gbdt_reranker(
        nrms=nrms,
        article_mapping=article_mapping,
        train_behaviors_raw=df_train_raw,
        df_val_evaluated=df_val_best,
        processed_dir=processed_dir,
        model_dir=model_dir,
        gbdt_train_samples=args.gbdt_train_samples,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
    )

    # Save comparison CSV
    comp_rows = []
    for m in ["AUC", "MRR", "nDCG@5", "nDCG@10"]:
        b_val = summary_nrms[m]
        a_val = summary_gbdt[m]
        diff = a_val - b_val
        pct = (diff / max(1e-9, b_val)) * 100.0
        comp_rows.append({
            "dataset": "ebnerd",
            "best_epoch": best_epoch,
            "metric": m,
            "before_nrms": round(b_val, 4),
            "after_gbdt": round(a_val, 4),
            "absolute_gain": round(diff, 4),
            "relative_gain_pct": round(pct, 2),
        })
    df_comp = pl.DataFrame(comp_rows)
    comp_csv = results_dir / "nrms_multiepoch_vs_gbdt.csv"
    df_comp.write_csv(comp_csv)
    print("\nFINAL MULTI-EPOCH NRMS vs GBDT RE-RANKER:")
    print(df_comp)


def main() -> None:
    args = parse_args()
    run_multiepoch_pipeline(args)


if __name__ == "__main__":
    main()
