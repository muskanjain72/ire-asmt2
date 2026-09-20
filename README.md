# CS4.406 Information Retrieval & Extraction — Assignment 2
## Two-Stage Retrieve-then-Rank News Recommendation on EB-NeRD & MIND

This repository contains the complete codebase, empirical experiments, technical reports, and deliverables for **Assignment 2: Learning from Click-Logs on EB-NeRD and MIND**.

---

## 📋 Deliverables Index (Q7 Compliance)

| Deliverable | Description | Primary Location |
|---|---|---|
| **1. Code & Pipelines** | Reproducible two-stage pipeline, GBDT re-ranker, NRMS baselines, feature stores, evaluation harness, test suite | [`ire_a1/recsys-ir/src/`](ire_a1/recsys-ir/src/), [`ire_a1/recsys-ir/scripts/`](ire_a1/recsys-ir/scripts/) |
| **2. Report (Moodle)** | 8-page Technical Design Note (`design_note.pdf`) & comprehensive technical markdown report | [`ire_a1/recsys-ir/report/design_note.pdf`](ire_a1/recsys-ir/report/design_note.pdf), [`a2.md`](a2.md) |
| **3. Leaderboard Screenshots** | Codabench competition leaderboard & submission verification screenshots for both MIND and EB-NeRD | [`ire_a1/recsys-ir/screenshots/`](ire_a1/recsys-ir/screenshots/) |
| **4. AI Usage Log** | Complete AI usage log (prompts, verbatim session exports 1–19, code attribution matrix) | [`logs/a2_ai_log.md`](logs/a2_ai_log.md), [`logs/session_exports/`](logs/session_exports/) |

---

## 🚀 Quick Start: One-Command Reproduce

### 1. Environment Setup
```bash
cd ire_a1/recsys-ir
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Run Test Suite (All 22 Unit & Anti-Leakage Tests)
```bash
pytest tests/ -v
```

### 3. One-Command Pipeline Reproduction
To run the full two-stage feature extraction, training, and evaluation pipeline:
```bash
# 1. Materialize feature stores and temporal behavioral signals
python -m src.feature_store.build_features --dataset all

# 2. Train the Stage-2 GBDT Re-Ranker on canonical protocol
python -m scripts.train_reranker --dataset ebnerd

# 3. Evaluate the Re-Ranker (AUC, MRR, nDCG@5, nDCG@10)
python -m scripts.eval_reranker --dataset ebnerd

# 4. Run the 6-group Ablation Study
python -m src.evaluation.ablation_study --dataset all

# 5. Run Paired Bootstrap Statistical Significance Testing (B=1,000)
python -m src.evaluation.bootstrap

# 6. Run Serving Latency, Memory & Scale Benchmark
python -m scripts.benchmark_serving
```

### 4. Codabench Competition Prediction Generation
```bash
# Generate EB-NeRD competition submission (GBDT Stage-2)
python -m scripts.generate_ebnerd_gbdt_submission

# Generate MIND competition submission (NRMS neural baseline)
python -m scripts.generate_mind_nrms_submission
```
*Note: Generated prediction zip files are excluded from Git per the assignment's `.gitignore` policy ("No large files — use .gitignore").*

---

## 📁 Repository Structure

```
├── a2.md                   # Exhaustive technical report & empirical findings
├── A2.pdf                  # Official course assignment specification
├── logs/
│   ├── a2_ai_log.md        # Official A2 AI usage log & code attribution
│   └── session_exports/    # 19 standalone markdown chat transcripts
└── ire_a1/recsys-ir/       # Core codebase & deliverables
    ├── report/             # Official Design Note deliverable
    │   ├── design_note.pdf # Compiled 8-page design note PDF for Moodle
    │   └── design_note.tex # LaTeX source
    ├── screenshots/        # Codabench leaderboard and submission screenshots
    │   ├── ebnerd-leaderboard1.png
    │   ├── ebnerd-subm1.png
    │   ├── ebnerd-subm2.png
    │   ├── mind-leaderboard1.png
    │   └── mind-subm1.png
    ├── src/
    │   ├── feature_store/  # User store, article store, behavioral features
    │   ├── behavioral/     # Position bias model, dwell/scroll dynamics
    │   ├── reranking/      # GBDT ranker, candidate vectorizer, pipeline
    │   ├── models/         # CategoryAwareNRMS architecture
    │   ├── evaluation/     # Metrics, paired bootstrap, ablation harness
    │   └── submission/     # Prediction formatting & packaging
    ├── scripts/            # Training, benchmarking, and submission scripts
    ├── tests/              # 22 passing unit, serving, and anti-leakage tests
    └── results/            # Canonical evaluation CSVs and metrics
```
