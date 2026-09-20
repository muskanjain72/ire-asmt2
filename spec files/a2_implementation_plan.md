# A2 Implementation Plan — `recsys-ir` Codebase

This plan is built strictly from verified file contents (directory tree, `article_store.py`,
`user_store.py` excerpts, `hybrid_rerank.py`, `store_backend.py`, `large_user_store.py`) — not
from `spec.md`'s unverified claims. Anywhere something couldn't be confirmed, it's marked
**UNVERIFIED** rather than assumed.

---

## 1. Verified Codebase State (evidence-based)

### ✅ Confirmed working and reusable as-is

| Component | Evidence | A2 relevance |
|---|---|---|
| DuckDB + Parquet feature store (`ParquetStore`) | `store_backend.py` — real DuckDB `:memory:` connection, `read_parquet()` view, parameterized SQL, columnar projection | Solves A1's scaling problem outright — no full-file loads needed |
| Memory-mapped history store for EB-NeRD-large | `large_user_store.py` imports `MemoryMappedHistoryStore` from `history_store.py` | Confirms the "mmap" claim, but scoped specifically to EB-NeRD's large-scale timestamped history — not a blanket claim across the whole system |
| Time-based exponential decay recency weighting | `user_store.py`: `weight = exp(-λ × Δt_seconds)`; `large_user_store.py`: identical formula, `DECAY_LAMBDA = 1/(7×86400)` (~7-day time constant) | **Q1's recency-weighting requirement is already done**, consistently at both small and large scale |
| Dataset-aware leakage handling, self-documented | `user_store.py` docstring: EB-NeRD strictly filters `clicked_at < as_of_ts`; MIND is pass-through with an explicit **"documented assumption, not a verified guarantee"** caveat | Strong foundation for Q9 — the honesty of this caveat should be quoted directly in the design note |
| A second, position-based recency scheme | `hybrid_rerank.py`: `recency_weighted_user_vector()` uses `decay ** (position_from_most_recent)`, decay=0.85, cap=20 — **different formula from `user_store.py`'s time-based decay** | **Inconsistency to resolve or explicitly justify** — see Section 5 |
| Heuristic hybrid candidate blending | `hybrid_rerank.py`: `hybrid_rank_candidates()` = `alpha·minmax(cosine_sim) + (1-alpha)·minmax(log1p(train_popularity))`, alpha=0.8 fixed | A real, working signal — but **not a trained re-ranker** (see Section 2) |
| Leakage-safe popularity signal | `hybrid_rerank.py` reuses `run_eval._load_popularity(dataset, "train", scale)` | Correctly train-split-only, consistent with A1's approach |
| Attention-ranker pipeline (scripts exist) | `scripts/train_attention_ranker.py`, `tune_query_key_attention.py`, `eval_attention_full.py`, `generate_attention_submission.py`, `results/large/qk_attention_tuning.csv` | Likely satisfies Q2's neural-ranker option — **architecture not yet reviewed (UNVERIFIED)** |
| Evaluation harness | `evaluation/`: `ranking_metrics.py`, `beyond_accuracy.py`, `bootstrap.py`, `slicing.py`, `run_eval.py`, `compare_retrievers.py`, `candidate_pool_sizes.py` | Matches A1's scope; `compare_retrievers.py` may already support before/after comparison — **contents not yet reviewed (UNVERIFIED)** |
| Existing design note draft | `design_note/design.md` referenced directly in `hybrid_rerank.py`'s docstring | **A design note already exists in some form — read this before starting Q6 from scratch** |

### ❌ Confirmed absent — must be built from scratch

- **Session features**: no `session_id`, `dwell_time`, or `position_bias` handling anywhere in the tree or in any file reviewed.
- **GBDT re-ranker**: no `lightgbm`/`xgboost`/`HistGradientBoostingClassifier` reference found; `scripts/train_reranker.py` confirmed **does not exist**.
- **Official baseline reproduction** (`ebnerd-benchmark`'s actual NRMS): no evidence anywhere of cloning or reproducing the official starter baseline. The attention scripts are custom-built, not a reproduction of the official one.
- **Category-affinity features**: not seen in any file reviewed (article store has `category`/`subcategory` columns, but no user-side affinity computation was found).
- **Serving/latency/memory benchmarking**: no `benchmark_serving.py` or equivalent found anywhere.
- **Paired bootstrap CI** (resampling the *same* impressions across two models to test whether their difference excludes zero): `bootstrap.py` exists but its resampling scheme is unverified — A1-style independent per-model CIs are a different, weaker statistical tool than what Q3 requires.

### ⚠️ Unverified — check before planning further

- `scripts/train_attention_ranker.py`'s actual architecture (is it a real NRMS-style attention model, or something simpler?)
- `compare_retrievers.py`'s actual comparison logic — may already do most of Q5's "before/after" reporting
- `bootstrap.py`'s actual resampling implementation — paired or independent?
- `design_note/design.md`'s current content — how much of Q6 is already drafted?
- `history_store.py`'s `MemoryMappedHistoryStore` implementation directly (currently only known by import, not by content)

---

## 2. Critical Framing Issue: What Actually Counts as "The Re-Ranker"?

This needs resolving with your teammate before writing anything new, since it changes the whole Q2/Q3 story:

- **`hybrid_rerank.py` is NOT a learned re-ranker.** It has no training/fitting step — `alpha=0.8` is a fixed constant, not something learned from click labels. Q2 explicitly asks to "train a re-ranker" (GBDT or neural). Using `hybrid_rerank.py` alone would not satisfy Q2's core requirement, even though it's a legitimate signal.
- **The attention scripts are the more likely real candidate for Q2's "Option B: small neural ranker."** But their architecture is currently unverified — this must be checked before assuming they satisfy the requirement.
- **Recommendation**: treat `hybrid_rerank.py`'s blended score as one additional *feature* fed into whatever the real trained ranker turns out to be (attention model, or a newly-built GBDT), not as the re-ranker itself. This also gives you a legitimate ablation axis for Q3 ("with vs. without the hybrid popularity signal as an input feature").

---

## 3. Task List by Assignment Question

### Q1 — Click-History & Session Features

| Task | Status | Action |
|---|---|---|
| Recency-weighted history | ✅ Done | None — reuse `user_store.py`/`large_user_store.py` as-is |
| Reconcile the two decay formulas | ⚠️ Open | Decide: keep both (document why — e.g. position-based for retrieval blending, time-based for feature-store recency), or unify into one. Must be explicitly justified in the design note either way, since a grader will notice two different decay mechanisms |
| Session features (dwell time, position bias) | ❌ Missing | New module, e.g. `src/feature_store/session_features.py`. EB-NeRD: derive from `session_id`, `read_time`, `scroll_percentage` (already in raw `behaviors.parquet`, confirmed in A1). MIND: no `session_id` — build a time-gap heuristic (e.g. impressions within N minutes of each other = one session) |
| Article freshness | ⚠️ Unverified | Check `article_store.py` fully (only first ~225 lines seen) for any `published_time`-based feature. EB-NeRD has `published_time`; MIND does not (confirmed in A1) — freshness may only be computable for EB-NeRD, and this limitation should be stated explicitly, not silently skipped |
| Category-affinity | ❌ Likely missing | New function: user's historical category distribution vs. candidate's category — build if not found in the unreviewed portion of `article_store.py` |
| Leakage boundary on all new features | Required | Every new feature function must accept `as_of_ts`/`before_timestamp` and be covered by an extension to `leakage_test.py` |

### Q2 — Re-Ranker

1. **Review `scripts/train_attention_ranker.py` in full** before deciding its role (this file's content still needs to be read — recommend doing this first, even though no more files were requested for this planning pass).
2. If it's a genuine trained architecture: treat it as your Q2 re-ranker, feed it the new Q1 features (session, freshness, category-affinity) plus `hybrid_rerank.py`'s blended score as an additional input feature.
3. If a GBDT path is wanted instead/additionally (simpler, likely faster for a 2-person team's remaining timeline): build `scripts/train_reranker.py` fresh using `sklearn.HistGradientBoostingClassifier` or LightGBM over the full engineered feature set, trained on real click labels.
4. Report AUC/MRR/nDCG **before** (raw retrieval order) and **after** (re-ranked) — check whether `compare_retrievers.py` already does this comparison pattern before building a new script.

### Q3 — Baseline Reproduced, Then Beaten

1. **Clone and reproduce the official `ebnerd-benchmark` NRMS baseline** — this is a hard gap, not present anywhere in the current codebase. Budget real time for this; it's the one piece that can't be adapted from existing code.
2. Establish the official baseline's numbers as the "before" ground truth.
3. Pick **one** principled improvement — likely candidate: the attention ranker (pending Section 2's review) or a newly-built GBDT with the richer Q1 feature set.
4. **Build a paired bootstrap CI script** — verify first whether `bootstrap.py` already supports paired resampling; if not, this needs new code: resample the *same* set of impressions for both models on each iteration, compute the metric difference per resample, and report whether the resulting CI excludes zero.

### Q4 — Serving & Scale Analysis (build entirely fresh)

1. `scripts/benchmark_serving.py`:
   - Memory: use `tracemalloc` or `psutil` to measure the actual loaded footprint of the ANN index + feature store (both `ParquetStore`-backed and `MemoryMappedHistoryStore`-backed paths, since they have different memory profiles worth comparing)
   - Latency: time N repeated single-user end-to-end requests (candidate generation → feature lookup → re-rank), discard warm-up runs, report p99
2. Cost/QPS back-of-envelope from the measured p99, against a stated target SLA (e.g., "p99 < 100ms")
3. 10x scaling argument — grounded in whichever component the real benchmark shows is the bottleneck (candidates worth checking specifically: DuckDB's per-query overhead under concurrent load, and whether the EB-NeRD mmap path or the MIND pass-through path scales differently)

### Q5 — Extended Evaluation

1. Review `compare_retrievers.py` and `candidate_pool_sizes.py` fully — these may already cover much of the "before/after" and slicing reporting needed; avoid duplicating them.
2. Run the full two-stage pipeline (retrieval → re-rank) at `DATA_SCALE=large`, confirming the DuckDB/mmap architecture holds up as claimed.
3. Confirm cold/warm and head/tail slicing (already present per the README) is applied to the **post-rerank** output, not just raw retrieval.

### Q6 — Design Note

**Read `design_note/design.md` first** — it's directly referenced as already existing. Don't start from a blank page; extend what's there with the new A2 sections: baseline-vs-improved with CI, serving/scale findings, the recency-decay-formula inconsistency (Section 5 here), and the 10x breakage analysis grounded in Q4's real measurements.

### Q9 — Anti-Gaming

1. Extend `leakage_test.py` to cover every new Q1 feature (session, freshness, category-affinity) — same `as_of_ts` boundary pattern as the existing tests.
2. Build the explicit "with vs. without serving-unavailable features" ablation A2 requires — identify any feature that could theoretically use future information (e.g., global popularity computed over the *full* dataset rather than train-only — note `hybrid_rerank.py` already gets this right by construction, good pattern to replicate for any new feature).

---

## 4. Suggested Order of Operations (2-person team)

1. **Both**: read `scripts/train_attention_ranker.py`, `compare_retrievers.py`, `bootstrap.py`, `design_note/design.md` in full — resolves every "UNVERIFIED" item above. Do this before dividing work, since it changes what's actually left to build.
2. **Person A**: Q1 session/freshness/category features + their leakage tests, in parallel with —
3. **Person B**: official NRMS baseline reproduction (the most externally-dependent, highest-risk-of-delay task — start it earliest)
4. **Both**: once both land, wire the chosen re-ranker (attention model and/or fresh GBDT) over the full feature set; resolve the decay-formula inconsistency explicitly
5. **Person A**: paired bootstrap CI script + ablation
6. **Person B**: serving/scale benchmarking script
7. **Both**: full-scale evaluation run, Codabench submissions, design note assembly

## 5. Open Decision to Make Before Coding Further

Two different recency-decay formulas currently coexist:
- `user_store.py` / `large_user_store.py`: **time-based**, `exp(-λ·Δt_seconds)`, ~7-day time constant
- `hybrid_rerank.py`: **position-based**, `decay^(steps_from_most_recent)`, decay=0.85, capped at last 20 items

These will produce different weightings for the same user (e.g., a user with a sparse history spanning months gets very different treatment under each scheme). Decide and document: are these serving genuinely different purposes (one for feature-store recency stats, one specifically for embedding-blend retrieval), or should they be unified? Either answer is defensible, but it must be a stated decision in the design note, not an unexamined inconsistency a grader discovers independently.
