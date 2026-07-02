# **Redrob AI Candidate Ranking System**

# Redrob AI Candidate Ranking System

## Overview

This repository contains the solution for the **Redrob AI Ranking Hackathon**.

The system discovers, filters, and ranks the **top 100 AI engineering candidates** from a dataset containing **100,000 candidates**.

To strictly comply with the **Stage 3 sandbox constraints** (offline execution, CPU-only, and a 5-minute wall-clock limit), the solution is divided into two phases:

1. **Offline Pre-computation (`precompute.py`)**

   * Performs all computationally expensive tasks such as parsing candidates, NLP processing, artifact generation, index construction, and model caching.
   * This phase has **no execution time limit**.

2. **Online Ranking (`rank.py`)**

   * Executes entirely offline using the precomputed artifacts.
   * Performs hybrid retrieval, mathematical gating, and localized cross-encoder reranking.
   * Completes well within the required **5-minute** execution limit.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Place the Required Data

Ensure the following files are placed in the repository root:

* `candidates.jsonl` (or `candidates.jsonl.gz`)
* `job_description.txt`

---

# Reproduction Steps

## Phase 1: Offline Pre-computation

This script performs the following tasks:

* Parses all 100K candidate profiles
* Computes behavioral and trajectory gating multipliers
* Builds the BM25 retrieval index
* Caches the Hugging Face cross-encoder model locally for completely offline execution during ranking

Run:

```bash
python precompute.py --candidates ./candidates.jsonl --outdir ./artifacts
```

### Generated Artifacts

The script creates the following files inside the `./artifacts` directory:

* `bm25_index.pkl`
* `candidate_metadata.pkl`
* `cross_encoder_model/`

These artifacts are **required** for the online ranking phase.

---

## Phase 2: Online Ranking (Stage 3 Reproduction)

This command generates the final submission file using the precomputed artifacts.

The ranking pipeline performs:

* Hybrid candidate retrieval
* Mathematical gating
* Deep semantic reranking
* Final selection of the Top 100 candidates

Run:

```bash
python rank.py \
    --candidates ./candidates.jsonl \
    --jd ./job_description.txt \
    --artifacts ./artifacts \
    --out ./submission.csv
```

---

## Output

After successful execution, the final ranked predictions will be written to:

```text
submission.csv
```

This CSV contains the final ranked list of the top 100 AI engineering candidates.

**Command:**  
python rank.py \--candidates ./candidates.jsonl \--jd ./job\_description.txt \--artifacts ./artifacts \--out ./submission.csv  
