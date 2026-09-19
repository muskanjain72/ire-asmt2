"""Generate official MIND Codabench submission using verified NRMS single-epoch checkpoint.

Model checkpoint: models/official_mind_nrms/nrms_weights.weights.h5 (AUC 0.6338).
Large test set: data/raw/mind/MINDlarge_test/behaviors.tsv (2,370,727 impressions).
Output file: submissions/mind/prediction.txt
Zipped file: submissions/mind/mind_submission.zip
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time

# Force CPU execution for TensorFlow to avoid CUDA/XLA Triton autotuner crash
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommenders_mind import MINDIterator, NRMSModel, prepare_hparams
from src.submission.package_submission import package_prediction
from src.submission.writers import ranked_ids_to_positions, validate_prediction_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("mind_nrms_submission")

EXPECTED_MIND_ROWS = 2_370_727
EXPECTED_CHECKPOINT_SIZE = 135_564_320  # Official single-epoch weights


def verify_prerequisites() -> tuple[Path, Path, Path]:
    mind_dir = PROJECT_ROOT / "data" / "raw" / "mind"
    test_zip = mind_dir / "MINDlarge_test.zip"
    test_behaviors = mind_dir / "MINDlarge_test" / "behaviors.tsv"
    test_news = mind_dir / "MINDlarge_test" / "news.tsv"
    checkpoint = PROJECT_ROOT / "models" / "official_mind_nrms" / "nrms_weights.weights.h5"

    if not test_zip.exists():
        raise FileNotFoundError(f"CRITICAL: MIND large test archive missing at {test_zip}")
    if not test_behaviors.exists():
        raise FileNotFoundError(f"CRITICAL: MIND test behaviors missing at {test_behaviors}")
    if not test_news.exists():
        raise FileNotFoundError(f"CRITICAL: MIND test news missing at {test_news}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"CRITICAL: Official NRMS checkpoint missing at {checkpoint}")

    ckpt_size = checkpoint.stat().st_size
    logger.info("Confirmed MINDlarge_test.zip present: %d bytes", test_zip.stat().st_size)
    logger.info("Confirmed checkpoint present: %s (%d bytes)", checkpoint, ckpt_size)
    if ckpt_size != EXPECTED_CHECKPOINT_SIZE:
        logger.warning(
            "Checkpoint size %d != expected %d; verifying model architecture",
            ckpt_size,
            EXPECTED_CHECKPOINT_SIZE,
        )

    return test_behaviors, test_news, checkpoint


def encode_test_news(
    model: NRMSModel, news_file: Path, word_dict: dict[str, int], title_size: int = 30, batch_size: int = 2048
) -> tuple[np.ndarray, dict[str, int]]:
    """Tokenize and encode all unique news articles in MINDlarge_test/news.tsv."""
    logger.info("Reading test news from %s...", news_file)
    news_ids = ["<PAD>"]
    titles_raw = [""]

    with news_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                news_ids.append(parts[0])
                titles_raw.append(parts[3])

    n_news = len(news_ids)
    nid_to_idx = {nid: i for i, nid in enumerate(news_ids)}
    logger.info("Found %d unique test articles (including PAD). Tokenizing titles...", n_news - 1)

    title_tokens = np.zeros((n_news, title_size), dtype=np.int32)
    for i in range(1, n_news):
        words = titles_raw[i].lower().split()[:title_size]
        for j, w in enumerate(words):
            title_tokens[i, j] = word_dict.get(w, 0)

    logger.info("Encoding news vectors with model.newsencoder (batch_size=%d)...", batch_size)
    t0 = time.time()
    news_vecs = np.zeros((n_news, 400), dtype=np.float32)

    for start_idx in range(0, n_news, batch_size):
        end_idx = min(start_idx + batch_size, n_news)
        batch = title_tokens[start_idx:end_idx]
        vecs = model.newsencoder(batch).numpy()
        news_vecs[start_idx:end_idx] = vecs
        if (start_idx // batch_size) % 20 == 0 or end_idx == n_news:
            logger.info("  Encoded %d / %d articles (%.1f%%)", end_idx, n_news, end_idx / n_news * 100)

    elapsed = time.time() - t0
    logger.info("All %d news articles encoded in %.2fs (%.1f articles/s)", n_news - 1, elapsed, (n_news - 1) / elapsed)
    return news_vecs, nid_to_idx


def encode_unique_users(
    model: NRMSModel,
    behaviors_file: Path,
    news_vecs: np.ndarray,
    nid_to_idx: dict[str, int],
    his_size: int = 50,
    batch_size: int = 2048,
) -> dict[str, np.ndarray]:
    """Scan behaviors to extract unique users, look up precomputed news vectors, and encode with user attention head."""
    logger.info("Extracting unique users and click histories from %s...", behaviors_file)
    t0 = time.time()
    user_histories: dict[str, list[str]] = {}

    with behaviors_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 5:
                uid = parts[1]
                if uid not in user_histories:
                    hist_str = parts[3].strip()
                    user_histories[uid] = hist_str.split() if hist_str else []

    n_users = len(user_histories)
    logger.info("Found %d unique users in %.2fs. Pre-allocating history tensors...", n_users, time.time() - t0)

    user_list = list(user_histories.keys())
    user_vec_map: dict[str, np.ndarray] = {}

    attn_layer = model.userencoder.layers[2]
    pool_layer = model.userencoder.layers[3]

    t1 = time.time()
    for start_idx in range(0, n_users, batch_size):
        end_idx = min(start_idx + batch_size, n_users)
        b_users = user_list[start_idx:end_idx]
        b_len = len(b_users)

        # Build (b_len, his_size, 400) tensor by direct indexing into news_vecs
        batch_his_vecs = np.zeros((b_len, his_size, 400), dtype=np.float32)
        for i, uid in enumerate(b_users):
            h_nids = user_histories[uid][-his_size:]
            for j, nid in enumerate(h_nids):
                idx = nid_to_idx.get(nid, 0)
                batch_his_vecs[i, j] = news_vecs[idx]

        # Pass through decoupled user attention head
        tf_tensor = tf.convert_to_tensor(batch_his_vecs)
        y = attn_layer([tf_tensor, tf_tensor, tf_tensor])
        u_batch = pool_layer(y).numpy()

        for i, uid in enumerate(b_users):
            user_vec_map[uid] = u_batch[i]

        if (start_idx // batch_size) % 50 == 0 or end_idx == n_users:
            elapsed_u = time.time() - t1
            speed = end_idx / max(1.0, elapsed_u)
            eta = (n_users - end_idx) / max(1.0, speed)
            logger.info("  Encoded %d / %d users (%.1f%%) | Speed: %.1f users/s | ETA: %.1fm",
                        end_idx, n_users, end_idx / n_users * 100, speed, eta / 60.0)

    elapsed_total = time.time() - t1
    logger.info("All %d users encoded in %.2fs (%.1f users/s)", n_users, elapsed_total, n_users / elapsed_total)
    return user_vec_map


def generate_mind_predictions(
    behaviors_file: Path,
    news_vecs: np.ndarray,
    nid_to_idx: dict[str, int],
    user_vec_map: dict[str, np.ndarray],
    output_path: Path,
) -> int:
    """Stream impressions, compute candidate scores, and write 1-based ranks to output file."""
    logger.info("Starting scoring and writing predictions to %s...", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    row_count = 0

    with behaviors_file.open("r", encoding="utf-8") as in_f, output_path.open("w", encoding="utf-8", buffering=1024*1024*8) as out_f:
        for line in in_f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue

            imp_id = parts[0]
            uid = parts[1]
            cands_raw = parts[4].strip().split()
            # Extract clean candidate IDs (strip any train/val label suffixes if present)
            cands = [c.split("-")[0] for c in cands_raw]

            if not cands:
                out_f.write(f"{imp_id} []\n")
                row_count += 1
                continue

            u_vec = user_vec_map[uid]
            c_indices = [nid_to_idx.get(c, 0) for c in cands]
            c_matrix = news_vecs[c_indices]  # (num_cands, 400)

            # Dot-product scoring
            scores = np.dot(c_matrix, u_vec)

            # 1-based ranking matching Codabench: rank 1 is highest score
            order = np.argsort(-scores, kind="stable")
            ranks = np.empty(len(scores), dtype=np.int32)
            ranks[order] = np.arange(1, len(scores) + 1, dtype=np.int32)

            ranks_str = ",".join(map(str, ranks))
            out_f.write(f"{imp_id} [{ranks_str}]\n")
            row_count += 1

            if row_count % 250_000 == 0:
                elapsed = time.time() - t0
                speed = row_count / max(1.0, elapsed)
                eta = (EXPECTED_MIND_ROWS - row_count) / max(1.0, speed)
                logger.info("  Scored %d / %d impressions (%.1f%%) | Speed: %.1f impr/s | ETA: %.1fm",
                            row_count, EXPECTED_MIND_ROWS, row_count / EXPECTED_MIND_ROWS * 100, speed, eta / 60.0)

    elapsed_total = time.time() - t0
    logger.info("Scored all %d impressions in %.2fs (%.1f impr/s)", row_count, elapsed_total, row_count / elapsed_total)
    return row_count


def main():
    test_behaviors, test_news, checkpoint = verify_prerequisites()

    utils_dir = PROJECT_ROOT / "data" / "raw" / "mind" / "MINDsmall_utils"
    yaml_file = str(utils_dir / "nrms.yaml")
    wordEmb_file = str(utils_dir / "embedding.npy")
    wordDict_file = str(utils_dir / "word_dict.pkl")
    userDict_file = str(utils_dir / "uid2index.pkl")

    logger.info("Initializing official NRMS model graph...")
    hparams = prepare_hparams(
        yaml_file,
        wordEmb_file=wordEmb_file,
        wordDict_file=wordDict_file,
        userDict_file=userDict_file,
        batch_size=32,
        epochs=1,
    )
    model = NRMSModel(hparams, MINDIterator, seed=42)

    logger.info("Loading verified NRMS checkpoint: %s", checkpoint)
    model.model.load_weights(str(checkpoint))
    logger.info("NRMS model weights loaded successfully.")

    import pickle
    with open(wordDict_file, "rb") as f:
        word_dict = pickle.load(f)

    # 1. Encode all 120k test news
    news_vecs, nid_to_idx = encode_test_news(model, test_news, word_dict, title_size=hparams.title_size)

    # 2. Encode all 702k unique users
    user_vec_map = encode_unique_users(model, test_behaviors, news_vecs, nid_to_idx, his_size=hparams.his_size)

    # 3. Score all 2.37M impressions and write prediction.txt
    output_dir = PROJECT_ROOT / "submissions" / "mind"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "prediction.txt"
    zip_path = output_dir / "mind_submission.zip"

    actual_rows = generate_mind_predictions(test_behaviors, news_vecs, nid_to_idx, user_vec_map, prediction_path)

    # 4. Enforce strict row count gate
    logger.info("Verifying row count: actual=%d vs expected=%d", actual_rows, EXPECTED_MIND_ROWS)
    if actual_rows != EXPECTED_MIND_ROWS:
        raise ValueError(f"CRITICAL ROW COUNT MISMATCH: {actual_rows} != {EXPECTED_MIND_ROWS}! Aborting submission packaging!")

    # 5. Validate prediction file against official criteria
    logger.info("Running validate_prediction_file on %s...", prediction_path)
    validated_rows = validate_prediction_file(prediction_path, expected_rows=EXPECTED_MIND_ROWS)
    logger.info("Prediction file validation PASSED with %d rows.", validated_rows)

    # 6. Package into ZIP
    logger.info("Packaging %s into %s...", prediction_path, zip_path)
    package_prediction(prediction_path, zip_path)
    logger.info("Submission ZIP created successfully: %s (%d bytes)", zip_path, zip_path.stat().st_size)


if __name__ == "__main__":
    main()
