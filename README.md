# ai-hackathon
# Intelligent Candidate Discovery Engine

This repository provides an automated, purely data-driven pipeline for ranking job candidates based on their career history, skills, and engagement metrics.

## Requirements

Install the dependencies before running the scripts:

```bash
pip install -r requirements.txt
```

## Workflow

The pipeline is split into two deterministic steps: **Precomputation** (generating artifacts from the dataset) and **Ranking** (scoring the pool against a specific job description).

### 1. Precomputation

Generate the BM25 index, gating metadata, and cross-encoder model. This step requires the raw candidates file (`.jsonl` or `.jsonl.gz`).

```bash
python precompute.py \
    --candidates candidates.jsonl.gz \
    --outdir artifacts/
```

### 2. Ranking

Score the candidates based on a specific Job Description (JD). This step uses the precomputed artifacts for lightning-fast retrieval and deep semantic reranking.

**Important**: The `--candidates` file is required for dataset compatibility validation (SHA-256 hash checking) to ensure the artifacts match the raw data.

```bash
python rank.py \
    --candidates candidates.jsonl.gz \
    --jd job_description.txt \
    --artifacts artifacts/ \
    --out submission.csv
```

## Reproducibility

- The pipeline uses strict fixed random seeds (NumPy, Python) to guarantee deterministic outputs.
- File operations dynamically support both `.jsonl` and `.jsonl.gz` datasets seamlessly.
- Artifacts are verified against the input candidates file via SHA-256 hash checks to prevent out-of-sync evaluations.

