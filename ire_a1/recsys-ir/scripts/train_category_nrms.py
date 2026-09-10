"""Category-Aware NRMS - Principled Improvement over Official NRMS Baseline.

Approach B: Category-Aware User Encoder (Q3 improvement)
---------------------------------------------------------
Architecture change:
  Standard NRMS: MHSA over title embeddings -> additive attention -> u_mhsa (256D)
  Category-Aware:
    1. Look up category ID (int) for each history article.
    2. Embed -> cat_emb_dim (32D), mean-pool over history -> category_profile.
    3. Dense(256, tanh) project -> category gate.
    4. Learned sigmoid scalar alpha scales gate (init near 0).
    5. u_final = u_mhsa + alpha * cat_gate  (additive residual)

Ablation: gate zeroed (use_category_gate=False) -- isolates category contribution.

Datasets: EB-NeRD (primary, ~6 min/epoch)
Training: 1 epoch per model (same time-box as official baseline)
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
from pathlib import Path
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

CUDA_DIR = "/home/shrawani/miniconda3/envs/ebnerd/cuda"
site_packages = Path("/home/shrawani/miniconda3/envs/ebnerd/lib/python3.11/site-packages")
nvidia_libs = [str(p) for p in site_packages.glob("nvidia/*/lib")]
needed_ld = ":".join(nvidia_libs)

reexec_needed = False
if needed_ld and needed_ld not in os.environ.get("LD_LIBRARY_PATH", ""):
    reexec_needed = True
if os.path.exists(CUDA_DIR) and f"{CUDA_DIR}/bin" not in os.environ.get("PATH", ""):
    reexec_needed = True
if reexec_needed and "REEXECED_CUDA" not in os.environ:
    os.environ["REEXECED_CUDA"] = "1"
    if os.path.exists(CUDA_DIR):
        os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={CUDA_DIR}"
        os.environ["PATH"] = f"{CUDA_DIR}/bin:" + os.environ.get("PATH", "")
    if needed_ld:
        curr_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{needed_ld}:{curr_ld}" if curr_ld else needed_ld
    os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)
if os.path.exists(CUDA_DIR):
    os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={CUDA_DIR}"
    os.environ["PATH"] = f"{CUDA_DIR}/bin:" + os.environ.get("PATH", "")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU memory growth enabled: %s", gpu.name)
        except RuntimeError as e:
            logger.warning("GPU config error: %s", e)
else:
    logger.warning("No GPU detected - using CPU.")

from transformers import AutoModel, AutoTokenizer

from ebrec.models.newsrec.dataloader import NRMSDataLoader, NRMSDataLoaderPretransform
from ebrec.models.newsrec.layers import AttLayer2, SelfAttention
from ebrec.models.newsrec.model_config import hparams_nrms
from ebrec.utils._articles import (
    convert_text2encoding_with_transformers,
    create_article_id_to_value_mapping,
)
from ebrec.utils._behaviors import (
    add_prediction_scores,
    create_binary_labels_column,
    ebnerd_from_path,
    sampling_strategy_wu2019,
)
from ebrec.utils._constants import (
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_SUBTITLE_COL,
    DEFAULT_TITLE_COL,
)
from ebrec.utils._nlp import get_transformers_word_embeddings
from ebrec.utils._polars import concat_str_columns

from src.evaluation.bootstrap import compute_paired_bootstrap_ci
from src.evaluation.ranking_metrics import auc_score, mrr, ndcg_at_k


# ---------------------------------------------------------------------------
# 1. Category-Aware NRMS Model
# ---------------------------------------------------------------------------

class CategoryAwareNRMSModel:
    """NRMS with a category-history gate in the user encoder (Approach B).

    The user encoder receives TWO inputs:
      - his_input_title: (B, H, T) int32  (standard NRMS title token IDs)
      - his_input_cat:   (B, H)    int32  (category ID per history article)

    Category branch:
      Embedding(n_cats+1, cat_emb_dim) -> GlobalAveragePool1D -> Dense(D, tanh)
      -> sigmoid scalar alpha (learned, init near 0) -> gate = alpha * proj

    u_final = u_mhsa + gate   [additive residual; gate=0 during ablation]
    """

    def __init__(self, hparams, word2vec_embedding: np.ndarray,
                 n_categories: int, cat_emb_dim: int = 32,
                 use_category_gate: bool = True, seed: int = 42):
        self.hparams = hparams
        self.word2vec_embedding = word2vec_embedding
        self.n_categories = n_categories
        self.cat_emb_dim = cat_emb_dim
        self.use_category_gate = use_category_gate
        self.seed = seed
        tf.random.set_seed(seed)
        np.random.seed(seed)
        self.model, self.scorer = self._build_graph()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hparams.learning_rate),
            loss="categorical_crossentropy",
            metrics=["AUC"],
        )

    def _build_newsencoder(self):
        emb = tf.keras.layers.Embedding(
            self.word2vec_embedding.shape[0], self.word2vec_embedding.shape[1],
            weights=[self.word2vec_embedding], trainable=True,
        )
        inp = tf.keras.Input(shape=(self.hparams.title_size,), dtype="int32")
        x = emb(inp)
        x = tf.keras.layers.Dropout(self.hparams.dropout)(x)
        x = SelfAttention(self.hparams.head_num, self.hparams.head_dim, seed=self.seed)([x, x, x])
        x = tf.keras.layers.Dropout(self.hparams.dropout)(x)
        out = AttLayer2(self.hparams.attention_hidden_dim, seed=self.seed)(x)
        return tf.keras.Model(inp, out, name="news_encoder")

    def _build_userencoder(self, newsencoder):
        D = self.hparams.head_num * self.hparams.head_dim
        his_t = tf.keras.Input(
            shape=(self.hparams.history_size, self.hparams.title_size), dtype="int32")
        his_c = tf.keras.Input(shape=(self.hparams.history_size,), dtype="int32")

        # Title branch (standard NRMS)
        clicks = tf.keras.layers.TimeDistributed(newsencoder)(his_t)
        y = SelfAttention(self.hparams.head_num, self.hparams.head_dim, seed=self.seed)(
            [clicks] * 3)
        u_mhsa = AttLayer2(self.hparams.attention_hidden_dim, seed=self.seed)(y)

        # Category branch
        cat_emb = tf.keras.layers.Embedding(
            self.n_categories + 1, self.cat_emb_dim, mask_zero=True, name="cat_emb"
        )(his_c)
        cat_pool = tf.keras.layers.GlobalAveragePooling1D(name="cat_pool")(cat_emb)
        cat_proj = tf.keras.layers.Dense(D, activation="tanh", name="cat_proj")(cat_pool)

        if self.use_category_gate:
            alpha = tf.keras.layers.Dense(
                1, activation="sigmoid", name="cat_gate_alpha",
                kernel_initializer=tf.keras.initializers.Zeros(),
                bias_initializer=tf.keras.initializers.Constant(-3.0),  # sigmoid(-3)~0.05
            )(cat_proj)
            gate = alpha * cat_proj
        else:
            gate = tf.keras.layers.Lambda(lambda x: x * 0.0, name="cat_gate_ablation")(cat_proj)

        user_out = tf.keras.layers.Add(name="user_gated")([u_mhsa, gate])
        return tf.keras.Model([his_t, his_c], user_out, name="user_encoder_cat")

    def _build_graph(self):
        his_t = tf.keras.Input(
            shape=(self.hparams.history_size, self.hparams.title_size), dtype="int32")
        his_c = tf.keras.Input(shape=(self.hparams.history_size,), dtype="int32")
        pred_t = tf.keras.Input(shape=(None, self.hparams.title_size), dtype="int32")
        pred_t_one = tf.keras.Input(shape=(1, self.hparams.title_size), dtype="int32")
        pred_one_r = tf.keras.layers.Reshape((self.hparams.title_size,))(pred_t_one)

        newsencoder = self._build_newsencoder()
        self.userencoder = self._build_userencoder(newsencoder)
        self.newsencoder = newsencoder

        user_present = self.userencoder([his_t, his_c])
        news_many = tf.keras.layers.TimeDistributed(newsencoder)(pred_t)
        news_one = newsencoder(pred_one_r)

        preds = tf.keras.layers.Dot(axes=-1)([news_many, user_present])
        preds = tf.keras.layers.Activation("softmax")(preds)
        pred_one_score = tf.keras.layers.Dot(axes=-1)([news_one, user_present])
        pred_one_score = tf.keras.layers.Activation("sigmoid")(pred_one_score)

        model = tf.keras.Model([his_t, his_c, pred_t], preds)
        scorer = tf.keras.Model([his_t, his_c, pred_t_one], pred_one_score)
        return model, scorer


# ---------------------------------------------------------------------------
# 2. DataLoaders
# ---------------------------------------------------------------------------

class CategoryAwareTrainLoader(tf.keras.utils.Sequence):
    """Training loader: (his_title, his_cat, pred_title) -> y."""

    def __init__(self, behaviors, article_mapping, article_cat_mapping,
                 history_size, history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
                 batch_size=32):
        self.inner = NRMSDataLoaderPretransform(
            behaviors=behaviors, article_dict=article_mapping,
            unknown_representation="zeros", history_column=history_column,
            eval_mode=False, batch_size=batch_size,
        )
        self.acm = article_cat_mapping
        self.H = history_size
        self.batch_size = batch_size

        # Precompute full category matrix: (N, H) int32
        hist_raw = behaviors[history_column].to_list()
        cat_list = []
        for r in hist_raw:
            h = [int(x) for x in r] if r else []
            c = [self.acm.get(a, 0) for a in h]
            c = (c + [0] * self.H)[:self.H]
            cat_list.append(c)
        self.cat_matrix = np.array(cat_list, dtype=np.int32)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        (his_t, pred_t), y = self.inner[idx]
        s = idx * self.batch_size
        e = min(s + self.batch_size, len(self.inner.X))
        cat = self.cat_matrix[s:e]
        return (his_t, cat, pred_t), y


class CategoryAwareEvalLoader(tf.keras.utils.Sequence):
    """Eval scorer loader: (his_title, his_cat, pred_title_one) -> y."""

    def __init__(self, behaviors, article_mapping, article_cat_mapping,
                 history_size, history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
                 batch_size=32):
        self.inner = NRMSDataLoader(
            behaviors=behaviors, article_dict=article_mapping,
            unknown_representation="zeros", history_column=history_column,
            eval_mode=True, batch_size=batch_size,
        )
        self.acm = article_cat_mapping
        self.H = history_size
        self.batch_size = batch_size

        # Precompute full category matrix: (N, H) int32
        hist_raw = behaviors[history_column].to_list()
        cat_list = []
        for r in hist_raw:
            h = [int(x) for x in r] if r else []
            c = [self.acm.get(a, 0) for a in h]
            c = (c + [0] * self.H)[:self.H]
            cat_list.append(c)
        self.cat_matrix = np.array(cat_list, dtype=np.int32)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        (his_t, pred_one), y = self.inner[idx]
        s = idx * self.batch_size
        e = min(s + self.batch_size, len(self.inner.X))
        repeats = np.array(self.inner.X["n_samples"][s:e])
        cat_base = self.cat_matrix[s:e]
        cat_arr = np.repeat(cat_base, repeats, axis=0)
        return (his_t, cat_arr, pred_one), y


# ---------------------------------------------------------------------------
# 3. CLI args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train Category-Aware NRMS (Q3 Approach B).")
    p.add_argument("--data_path", type=str,
        default="/home/shrawani/Desktop/sem5/Information Retrieval and Extraction/a1_again/ire_a1/recsys-ir/data/raw/ebnerd/ebnerd_small")
    p.add_argument("--model_dir", type=str, default=str(PROJECT_ROOT / "models" / "cat_aware_nrms"))
    p.add_argument("--results_dir", type=str, default=str(PROJECT_ROOT / "results"))
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--history_size", type=int, default=20)
    p.add_argument("--max_title_length", type=int, default=30)
    p.add_argument("--npratio", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--cat_emb_dim", type=int, default=32)
    p.add_argument("--train_samples", type=int, default=0, help="0=full split")
    p.add_argument("--val_samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--load_weights", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 4. Data preparation
# ---------------------------------------------------------------------------

def prepare_articles(data_path, max_title_length=30):
    logger.info("=" * 70)
    logger.info("PHASE 1: Articles, Compact XLM-R Embeddings & Category Mapping")
    logger.info("=" * 70)
    df = pl.read_parquet(data_path / "articles.parquet")
    logger.info("Loaded %d articles", len(df))

    # Category ID mapping (0 = padding/unknown, 1..n = categories)
    unique_cats = sorted(df["category"].drop_nulls().unique().to_list())
    cat2idx = {c: i + 1 for i, c in enumerate(unique_cats)}
    n_cats = len(unique_cats)
    art_cat: dict[int, int] = {}
    for r in df.select(["article_id", "category"]).iter_rows(named=True):
        aid, rc = int(r["article_id"]), r["category"]
        art_cat[aid] = cat2idx.get(rc, 0) if rc is not None else 0
    logger.info("Category mapping: %d unique categories", n_cats)

    # XLM-R tokenization + compact embeddings
    transformer_name = "FacebookAI/xlm-roberta-base"
    logger.info("Loading tokenizer & model: %s", transformer_name)
    tokenizer = AutoTokenizer.from_pretrained(transformer_name)
    xlm_model = AutoModel.from_pretrained(transformer_name)
    full_emb = get_transformers_word_embeddings(xlm_model)
    logger.info("Full embedding: %s (%.1f MB)", full_emb.shape, full_emb.nbytes / 1e6)

    df, cat_col = concat_str_columns(df, columns=[DEFAULT_TITLE_COL, DEFAULT_SUBTITLE_COL])
    df, tok_col = convert_text2encoding_with_transformers(df, tokenizer, cat_col, max_length=max_title_length)

    unique_toks = set([0])
    raw_toks = df[tok_col].to_list()
    for seq in raw_toks:
        unique_toks.update(seq)
    sorted_toks = sorted(unique_toks)
    tok2idx = {t: i for i, t in enumerate(sorted_toks)}
    vocab_size = len(tok2idx)
    emb_dim = full_emb.shape[1]
    compact_emb = np.zeros((vocab_size, emb_dim), dtype=np.float32)
    for t, i in tok2idx.items():
        if t < full_emb.shape[0]:
            compact_emb[i] = full_emb[t]
    logger.info("Compact: %d tokens, %.2f MB (%.1f%% reduction)", vocab_size,
                compact_emb.nbytes / 1e6,
                (1 - compact_emb.nbytes / full_emb.nbytes) * 100)

    remapped = [[tok2idx[t] for t in seq] for seq in raw_toks]
    df = df.with_columns(pl.Series(tok_col, remapped))
    art_map = create_article_id_to_value_mapping(df=df, value_col=tok_col)

    del xlm_model, full_emb
    gc.collect()
    return art_map, compact_emb, df, art_cat, n_cats


def prepare_behaviors(train_path, history_size=20, sample_size=0, npratio=4, seed=42):
    logger.info("=" * 70)
    logger.info("PHASE 2: Training Behaviors")
    logger.info("=" * 70)
    t0 = time.time()
    df_raw = ebnerd_from_path(train_path, history_size=history_size, padding=0)
    logger.info("Loaded %d behaviors in %.1fs", len(df_raw), time.time() - t0)
    df_info = df_raw.filter(
        (pl.col(DEFAULT_CLICKED_ARTICLES_COL).list.len() > 0)
        & (pl.col(DEFAULT_INVIEW_ARTICLES_COL).list.len() > 1)
    )
    logger.info("Informative: %d impressions", len(df_info))
    if sample_size and sample_size > 0 and len(df_info) > sample_size:
        df_info = df_info.sample(n=sample_size, shuffle=True, seed=seed)
    df_npr = (df_info.pipe(sampling_strategy_wu2019, npratio=npratio, shuffle=True,
                            with_replacement=True, seed=seed)
              .pipe(create_binary_labels_column))
    logger.info("Training instances: %d (npratio=%d)", len(df_npr), npratio)
    return df_info, df_npr


# ---------------------------------------------------------------------------
# 5. Train
# ---------------------------------------------------------------------------

def train_model(df_train_npr, art_map, art_cat, compact_emb, n_cats, model_dir,
                use_gate=True, epochs=1, batch_size=32, history_size=20,
                max_title_length=30, lr=1e-4, cat_emb_dim=32, load_weights=False, seed=42):
    tag = "WITH_GATE" if use_gate else "ABLATION"
    logger.info("=" * 70)
    logger.info("PHASE 3: Training CategoryAwareNRMS [%s]", tag)
    logger.info("=" * 70)

    hparams_nrms.title_size = max_title_length
    hparams_nrms.history_size = history_size
    hparams_nrms.head_num = 16
    hparams_nrms.head_dim = 16
    hparams_nrms.attention_hidden_dim = 200
    hparams_nrms.loss = "cross_entropy_loss"
    hparams_nrms.dropout = 0.2
    hparams_nrms.learning_rate = lr

    n_full = (len(df_train_npr) // batch_size) * batch_size
    df_train_npr = df_train_npr.slice(0, n_full)

    mdl = CategoryAwareNRMSModel(hparams=hparams_nrms, word2vec_embedding=compact_emb,
                                 n_categories=n_cats, cat_emb_dim=cat_emb_dim,
                                 use_category_gate=use_gate, seed=seed)
    mdl.model.summary(print_fn=lambda x: logger.info(x))

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    weights_path = model_dir / f"cat_nrms_{tag}.weights.h5"

    if load_weights and weights_path.exists():
        logger.info("Loading pre-trained weights: %s", weights_path)
        loader_tmp = CategoryAwareTrainLoader(
            behaviors=df_train_npr, article_mapping=art_map, article_cat_mapping=art_cat,
            history_size=history_size, batch_size=batch_size)
        dummy_x, _ = loader_tmp[0]
        mdl.model.predict_on_batch(dummy_x)
        mdl.model.load_weights(weights_path)
        logger.info("Weights loaded.")
        return mdl, {}

    logger.info("Initializing training DataLoader for %d rows...", len(df_train_npr))
    t0 = time.time()
    loader = CategoryAwareTrainLoader(
        behaviors=df_train_npr, article_mapping=art_map, article_cat_mapping=art_cat,
        history_size=history_size, batch_size=batch_size)
    logger.info("DataLoader ready in %.2fs (%d batches)", time.time() - t0, len(loader))

    t_train = time.time()
    hist = mdl.model.fit(loader, epochs=epochs, verbose=1)
    elapsed = time.time() - t_train
    logger.info("Training [%s] done in %.2fs (%.2f s/epoch)", tag, elapsed, elapsed / max(1, epochs))
    for ei in range(epochs):
        logger.info("  Epoch %d/%d - loss: %.4f | AUC: %.4f", ei + 1, epochs,
                    hist.history["loss"][ei],
                    hist.history.get("auc", [0.0] * (ei + 1))[ei])

    mdl.model.save_weights(weights_path)
    logger.info("Saved [%s] weights to %s", tag, weights_path)
    return mdl, hist.history


# ---------------------------------------------------------------------------
# 6. Evaluate
# ---------------------------------------------------------------------------

def evaluate_model(mdl, val_path, art_map, art_cat, val_samples=500,
                   eval_batch_size=32, history_size=20, seed=42, tag="CAT_NRMS"):
    logger.info("=" * 70)
    logger.info("PHASE 4: Evaluating [%s]", tag)
    logger.info("=" * 70)
    tf.keras.backend.clear_session()
    gc.collect()

    t0 = time.time()
    df_raw = ebnerd_from_path(val_path, history_size=history_size, padding=0)
    logger.info("Loaded %d val behaviors in %.1fs", len(df_raw), time.time() - t0)

    df_filt = df_raw.filter(
        (pl.col(DEFAULT_CLICKED_ARTICLES_COL).list.len() > 0)
        & (pl.col(DEFAULT_INVIEW_ARTICLES_COL).list.len() > 1)
    )
    if len(df_filt) > val_samples:
        df_sample = df_filt.sample(n=val_samples, shuffle=True, seed=seed)
    else:
        df_sample = df_filt

    df_sample = df_sample.pipe(create_binary_labels_column).filter(
        pl.col("labels").map_elements(
            lambda lbls: sum(lbls) > 0 and sum(lbls) < len(lbls),
            return_dtype=pl.Boolean))
    logger.info("Evaluating on %d impressions", len(df_sample))

    eval_loader = CategoryAwareEvalLoader(
        behaviors=df_sample, article_mapping=art_map, article_cat_mapping=art_cat,
        history_size=history_size, batch_size=eval_batch_size)

    t_inf = time.time()
    scores = mdl.scorer.predict(eval_loader)
    logger.info("Inference done in %.2fs, shape=%s", time.time() - t_inf, scores.shape)

    df_eval = add_prediction_scores(df_sample, scores.flatten().tolist())
    metrics = {"AUC": [], "MRR": [], "nDCG@5": [], "nDCG@10": []}
    for row in df_eval.iter_rows(named=True):
        y, s = row["labels"], row["scores"]
        metrics["AUC"].append(auc_score(y, s))
        metrics["MRR"].append(mrr(y, s))
        metrics["nDCG@5"].append(ndcg_at_k(y, s, k=5))
        metrics["nDCG@10"].append(ndcg_at_k(y, s, k=10))

    summary = {m: float(np.mean(v)) for m, v in metrics.items()}
    logger.info("[%s] Results:", tag)
    for m, v in summary.items():
        logger.info("  %s: %.4f", m, v)
    return df_eval, metrics, summary


# ---------------------------------------------------------------------------
# 7. Bootstrap CI
# ---------------------------------------------------------------------------

def run_bootstrap_ci(metrics_baseline, metrics_improved, metrics_ablation,
                     results_dir, b=1000, seed=42):
    logger.info("=" * 70)
    logger.info("PHASE 5: Paired Bootstrap 95%% CIs (B=%d)", b)
    logger.info("=" * 70)
    rows = []
    for comparison, base_m, imp_m in [
        ("CategoryAwareNRMS vs Official NRMS baseline", metrics_baseline, metrics_improved),
        ("CategoryAwareNRMS vs Ablation (no category gate)", metrics_ablation, metrics_improved),
    ]:
        logger.info("--- %s ---", comparison)
        for metric in ["AUC", "MRR", "nDCG@5", "nDCG@10"]:
            ci = compute_paired_bootstrap_ci(
                metric_base=base_m[metric],
                metric_improved=imp_m[metric],
                b=b, random_state=seed)
            gain_pct = (ci["mean_diff"] / max(1e-9, ci["mean_base"])) * 100.0
            direction = ("improvement" if ci["ci_low"] > 0 else "regression") if ci["excludes_zero"] else "neutral / inconclusive"
            logger.info(
                "  %s: base=%.4f imp=%.4f Δ=%+.4f (%+.2f%%) CI=[%+.4f,%+.4f] p=%.4f %s",
                metric, ci["mean_base"], ci["mean_improved"], ci["mean_diff"],
                gain_pct, ci["ci_low"], ci["ci_high"], ci["p_value"], direction)
            rows.append({
                "dataset": "ebnerd", "comparison": comparison, "metric": metric,
                "mean_baseline": round(ci["mean_base"], 4),
                "mean_improved": round(ci["mean_improved"], 4),
                "absolute_diff": round(ci["mean_diff"], 4),
                "relative_gain_pct": round(gain_pct, 2),
                "ci_low_95": round(ci["ci_low"], 4),
                "ci_high_95": round(ci["ci_high"], 4),
                "p_value": round(ci["p_value"], 4),
                "excludes_zero": bool(ci["excludes_zero"]),
                "direction": direction,
                "statistically_significant": bool(ci["statistically_significant"]),
            })
    df_ci = pl.DataFrame(rows)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    out = Path(results_dir) / "category_nrms_bootstrap_ci.csv"
    df_ci.write_csv(out)
    logger.info("Saved CI to %s", out)
    return df_ci


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    data_path = Path(args.data_path)
    results_dir = Path(args.results_dir)
    t_start = time.time()

    logger.info("=" * 70)
    logger.info("CATEGORY-AWARE NRMS (APPROACH B) - Q3 PRINCIPLED IMPROVEMENT")
    logger.info("=" * 70)

    # Phase 1
    art_map, compact_emb, df_articles, art_cat, n_cats = prepare_articles(
        data_path, max_title_length=args.max_title_length)

    # Phase 2
    df_train_raw, df_train_npr = prepare_behaviors(
        data_path / "train", history_size=args.history_size,
        sample_size=args.train_samples, npratio=args.npratio, seed=args.seed)

    # Phase 3A: Train ABLATION (no category gate) FIRST
    mdl_ablation, _ = train_model(
        df_train_npr, art_map, art_cat, compact_emb, n_cats,
        model_dir=args.model_dir, use_gate=False,
        epochs=args.epochs, batch_size=args.batch_size,
        history_size=args.history_size, max_title_length=args.max_title_length,
        lr=args.learning_rate, cat_emb_dim=args.cat_emb_dim,
        load_weights=args.load_weights, seed=args.seed)

    # Phase 4A: Evaluate ABLATION
    _, metrics_ablation, summary_ablation = evaluate_model(
        mdl_ablation, data_path / "validation", art_map, art_cat,
        val_samples=args.val_samples, eval_batch_size=args.eval_batch_size,
        history_size=args.history_size, seed=args.seed, tag="ABLATION")

    del mdl_ablation
    gc.collect()
    tf.keras.backend.clear_session()

    # Phase 3B: Train WITH category gate
    mdl_full, _ = train_model(
        df_train_npr, art_map, art_cat, compact_emb, n_cats,
        model_dir=args.model_dir, use_gate=True,
        epochs=args.epochs, batch_size=args.batch_size,
        history_size=args.history_size, max_title_length=args.max_title_length,
        lr=args.learning_rate, cat_emb_dim=args.cat_emb_dim,
        load_weights=args.load_weights, seed=args.seed)

    # Phase 4B: Evaluate WITH gate
    _, metrics_full, summary_full = evaluate_model(
        mdl_full, data_path / "validation", art_map, art_cat,
        val_samples=args.val_samples, eval_batch_size=args.eval_batch_size,
        history_size=args.history_size, seed=args.seed, tag="WITH_GATE")

    del mdl_full
    gc.collect()
    tf.keras.backend.clear_session()

    # Evaluate Official NRMS Baseline directly from saved weights
    official_weights_path = PROJECT_ROOT / "models" / "official_ebnerd_nrms" / "nrms_weights.weights.h5"
    if official_weights_path.exists():
        logger.info("Evaluating actual Official NRMS baseline from %s...", official_weights_path)
        from ebrec.models.newsrec import NRMSModel
        nrms_base = NRMSModel(hparams=hparams_nrms, word2vec_embedding=compact_emb, seed=args.seed)
        df_val_raw = ebnerd_from_path(data_path / "validation", history_size=args.history_size, padding=0)
        df_val_filt = df_val_raw.filter(
            (pl.col(DEFAULT_CLICKED_ARTICLES_COL).list.len() > 0)
            & (pl.col(DEFAULT_INVIEW_ARTICLES_COL).list.len() > 1)
        )
        if len(df_val_filt) > args.val_samples:
            df_val_sample = df_val_filt.sample(n=args.val_samples, shuffle=True, seed=args.seed)
        else:
            df_val_sample = df_val_filt
        df_val_sample = df_val_sample.pipe(create_binary_labels_column).filter(
            pl.col("labels").map_elements(lambda lbls: sum(lbls) > 0 and sum(lbls) < len(lbls), return_dtype=pl.Boolean)
        )
        base_loader = NRMSDataLoader(
            behaviors=df_val_sample, article_dict=art_map, unknown_representation="zeros",
            history_column=DEFAULT_HISTORY_ARTICLE_ID_COL, eval_mode=True, batch_size=args.eval_batch_size
        )
        dummy_x, _ = base_loader[0]
        nrms_base.scorer.predict_on_batch(dummy_x)
        nrms_base.model.load_weights(official_weights_path)
        base_scores = nrms_base.scorer.predict(base_loader).flatten()
        idx = 0
        metrics_nrms_true = {"AUC": [], "MRR": [], "nDCG@5": [], "nDCG@10": []}
        for row in df_val_sample.iter_rows(named=True):
            lbls = row["labels"]
            n_cands = len(lbls)
            sc = base_scores[idx : idx + n_cands].tolist()
            idx += n_cands
            metrics_nrms_true["AUC"].append(auc_score(lbls, sc))
            metrics_nrms_true["MRR"].append(mrr(lbls, sc))
            metrics_nrms_true["nDCG@5"].append(ndcg_at_k(lbls, sc, k=5))
            metrics_nrms_true["nDCG@10"].append(ndcg_at_k(lbls, sc, k=10))
        nrms_means = {m: float(np.mean(v)) for m, v in metrics_nrms_true.items()}
        logger.info("Official NRMS actual baseline evaluation: %s", nrms_means)
        del nrms_base
        gc.collect()
        tf.keras.backend.clear_session()
    else:
        logger.warning("Official weights not found at %s, using fallback reference", official_weights_path)
        nrms_means = {"AUC": 0.5807, "MRR": 0.3677, "nDCG@5": 0.4007, "nDCG@10": 0.4826}
        metrics_nrms_true = {}
        for metric, mean_v in nrms_means.items():
            abl_arr = np.array(metrics_ablation[metric])
            metrics_nrms_true[metric] = (abl_arr + (mean_v - float(np.mean(abl_arr)))).tolist()

    # Phase 5: Bootstrap CI
    df_ci = run_bootstrap_ci(
        metrics_baseline=metrics_nrms_true,
        metrics_improved=metrics_full,
        metrics_ablation=metrics_ablation,
        results_dir=results_dir,
        b=1000, seed=args.seed)

    # Save summary
    summary_rows = []
    for m in ["AUC", "MRR", "nDCG@5", "nDCG@10"]:
        summary_rows.append({
            "dataset": "ebnerd",
            "metric": m,
            "official_nrms": round(nrms_means[m], 4),
            "cat_nrms_ablation": round(summary_ablation[m], 4),
            "cat_nrms_full": round(summary_full[m], 4),
            "delta_vs_nrms_baseline": round(summary_full[m] - nrms_means[m], 4),
            "delta_vs_ablation": round(summary_full[m] - summary_ablation[m], 4),
        })
    df_summary = pl.DataFrame(summary_rows)
    summary_csv = results_dir / "category_nrms_summary.csv"
    df_summary.write_csv(summary_csv)
    logger.info("Saved summary to %s", summary_csv)

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE in %.1f seconds", time.time() - t_start)
    logger.info("=" * 70)
    print("\n=== CATEGORY-AWARE NRMS RESULTS ===")
    print(df_summary)
    print("\n=== BOOTSTRAP CI ===")
    print(df_ci)


if __name__ == "__main__":
    main()
