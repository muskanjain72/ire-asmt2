# A2 Codebase Analysis & Specification

## 1. Analysis against A2.pdf Requirements

Assignment 2 (A2) builds upon A1 and requires the following key components:
1. **Click-History & Session Features:** Behavioral signals, recency-weighting, session features (dwell time, position bias), and strict leakage boundaries.
2. **Re-Ranker:** Two-stage retrieve-then-rank using GBDT or a small neural ranker (like NRMS).
3. **Baseline & Ablation:** Reproduce an official baseline (e.g., NRMS), improve it, and prove gains with a paired bootstrap 95% CI.
4. **Serving & Scale Analysis:** Index memory footprint, p99 latency, cost/QPS SLA, and a 10× scaling argument.
5. **Extended Evaluation:** Full offline metrics with slicing (cold/warm, head/tail) on large-scale datasets.

### Codebase 1: `ire-asmt1`
- **State:** A baseline, clean Assignment 1 implementation.
- **Strengths:** Implements basic BM25 and semantic (MiniLM) candidate generation. It features a clean evaluation harness and leakage tests.
- **Weaknesses:** It is structurally naive. The README explicitly states that it breaks at large scale (doing full-corpus ranking in Python loops), which is a blocker since A2 requires processing the `large` Codabench datasets. It completely lacks any re-ranking logic, behavioral features, or advanced user representations.

### Codebase 2: `ire_a1/recsys-ir`
- **State:** An advanced, highly optimized pipeline that has already tackled the hard scaling problems and implemented early versions of A2 components.
- **Strengths:**
  - **Scale:** Utilizes DuckDB and memory-mapped (`mmap`) NumPy arrays to handle millions of rows without OOM (out-of-memory) errors, perfectly aligning with A2's scale requirements.
  - **Re-Rankers:** Already contains implementations for a GBDT combiner (`HistGradientBoostingClassifier`), a hybrid popularity ranker, and an NRMS-lite attention-based neural ranker (`train_attention_ranker.py`).
  - **Features:** Already implements recency-weighted user vectors and category-affinity features.
  - **Evaluation:** Has a robust test suite for leakage and implements bootstrap CI offline evaluation.
- **Missing Pieces for A2:**
  - **Session Features:** It lacks explicit session-level features like dwell time and position bias.
  - **Official Baseline Reproduction:** It implements an "NRMS-lite" on frozen mpnet embeddings, but A2 requires reproducing the *official/starter baseline* (e.g., fine-tuned NRMS from the ebnerd-benchmark repo) as the comparison point.
  - **Formal Serving Metrics:** While it deeply analyzes throughput (`ms/imp`), it needs a formal measurement script for **p99 latency**, strict **index memory** profiling, and a **cost/QPS estimate** to fulfill Q4.
  - **Paired CI Ablation:** Needs a dedicated script to output the paired bootstrap 95% CI specifically isolating the proposed re-ranker improvement over the baseline.

---

## 2. Decision

**Continue with `ire_a1/recsys-ir`**

*Why?* `ire_a1` is vastly more faithful to the physical realities of A2. A2 mandates evaluating on the `large` datasets (13.5M EB-NeRD impressions / 2.37M MIND impressions). `ire-asmt1` will crash with OOM on these datasets. `ire_a1/recsys-ir` has already solved this engineering bottleneck using `mmap` and DuckDB. Furthermore, it already contains ~80% of the A2 machine learning requirements (neural rankers, recency weighting, category affinity), giving us a massive head start on writing the final report.

---

## 3. Implementation Steps for A2

To complete A2 using `ire_a1/recsys-ir`, follow these steps:

### Step 1: Engineer Session & Dwell Features (Q1)
- Extract within-session click patterns, dwell time, and position bias from the behaviors Parquet.
- Integrate these into the DuckDB feature store and expose them to the candidate generators.

### Step 2: Reproduce the Official Baseline (Q3)
- Integrate or strictly mimic the official NRMS baseline from the `ebnerd-benchmark` repository for both MIND and EB-NeRD.
- Ensure the offline evaluation harness can score this baseline to establish the "before" metrics.

### Step 3: Finalize Re-Ranker & Ablation (Q2 & Q3)
- Choose one of the existing advanced rankers (e.g., the QK-attention or NRMS-lite ranker) as the "principled improvement".
- Run a strict ablation study isolating this change.
- Implement a paired bootstrap 95% CI script comparing the improvement against the official baseline to prove statistical significance (excluding zero).

### Step 4: Serving & Scale Analysis (Q4)
- Write a benchmarking script (`scripts/benchmark_serving.py`) that loads the index and measures exact **p99 retrieval latency** for single-user requests.
- Profile the exact memory footprint of the active ANN index + feature store in RAM.
- Calculate the back-of-the-envelope Cost/QPS based on cloud free-tier GPU specs.
- Draft the 10× scaling argument based on the existing mmap bottlenecks.

### Step 5: Extended Evaluation & Design Note (Q5 & Q6)
- Run the full pipeline (`make eval DATA_SCALE=large`) ensuring cold-start vs. warm and head vs. tail slices are reported.
- Generate final predictions and verify Codabench submission formats.
- Write the 6-page design note summarizing the architecture, baseline comparison, and serving analysis.
