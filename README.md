# **Redrob AI Candidate Ranking System**

## **Overview**

This repository contains the solution for the Redrob AI ranking hackathon. The system discovers, filters, and ranks the top 100 AI engineering candidates out of a 100,000-candidate dataset.  
To strictly comply with the Stage 3 sandbox constraints (offline execution, 5-minute wall-clock limit, CPU-only), the architecture is bifurcated into two phases:

1. **Offline Pre-computation (precompute.py):** Handles the heavy NLP, parsing, artifact generation, and model caching. No time limit applies here.  
2. **Online Ranking (rank.py):** Executes high-speed matrix math, gating logic, and localized cross-encoder reranking entirely offline in under 5 minutes.

## **Setup Instructions**

1. **Clone the repository:**  
   git clone \<your-repo-url\>  
   cd \<your-repo-directory\>

2. **Install Dependencies:**  
   pip install \-r requirements.txt

3. **Data Placement:**  
   Ensure your candidate dataset (candidates.jsonl or candidates.jsonl.gz) and the Job Description text file (job\_description.txt) are placed in the repository root.

## **Reproduction Steps**

### **Phase 1: Pre-computation (Offline Artifact Generation)**

This script parses the 100K candidates, computes all behavioral/trajectory gating multipliers, builds the BM25 index, and caches the Hugging Face cross-encoder model locally so Phase 2 can execute without network access.  
**Command:**  
python precompute.py \--candidates ./candidates.jsonl \--outdir ./artifacts

*(Note: This step generates bm25\_index.pkl, candidate\_metadata.pkl, and a local cross\_encoder\_model/ directory inside the ./artifacts folder. These artifacts are strictly required for Phase 2).*

### **Phase 2: Online Ranking (Stage 3 Reproduction)**

This is the **single command** that produces the final submission CSV from the candidates file and the pre-computed artifacts. It executes hybrid retrieval, mathematical gating, and deep semantic reranking.  
**Command:**  
python rank.py \--candidates ./candidates.jsonl \--jd ./job\_description.txt \--artifacts ./artifacts \--out ./submission.csv  
