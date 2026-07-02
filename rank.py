import os
import csv
import sys
import time
import pickle
import argparse
import hashlib
from pathlib import Path

# Deterministic execution
import random
import numpy as np

random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        print(f"ERROR: Failed to hash file {filepath}. Details: {e}")
        sys.exit(1)

def build_retrieval_query(jd_text: str) -> str:
    jd_lower = jd_text.lower()
    
    tech_keywords = [
        "machine learning", "ai", "artificial intelligence", "data science",
        "search", "ranking", "recommendation", "recommender",
        "embeddings", "vector", "retrieval", "nlp", "deep learning",
        "python", "pytorch", "tensorflow", "spark", "hadoop", "java", "c++", "go",
        "faiss", "qdrant", "pinecone", "weaviate", "milvus",
        "elasticsearch", "solr", "lucene", "lambdamart",
        "ndcg", "mrr", "map", "a/b test", "ab test", "evaluation",
        "production", "deployment", "api", "microservice",
        "software engineer", "backend", "distributed", "infrastructure"
    ]
    
    extracted = [kw for kw in tech_keywords if kw in jd_lower]
            
    synonyms = {
        "embedding": ["sentence-transformers", "bge", "e5", "embeddings"],
        "vector database": ["faiss", "pinecone", "weaviate", "milvus", "qdrant"],
        "learning-to-rank": ["lambdamart", "xgboost", "lightgbm", "ltr"],
        "evaluation": ["ndcg", "mrr", "map"],
        "retrieval": ["search", "ranking", "recommendation"]
    }
    
    for concept, expansions in synonyms.items():
        if concept in jd_lower:
            extracted.extend(expansions)
            
    query = " ".join(set(extracted))
    return query if query else jd_lower

def main():
    parser = argparse.ArgumentParser(description="Rank candidates based on precomputed artifacts.")
    parser.add_argument("--candidates", type=str, required=True, help="Path to the original candidates JSONL/GZ file (for validation).")
    parser.add_argument("--jd", type=str, default=None, help="Path to the job description text file (optional).")
    parser.add_argument("--artifacts", type=str, required=True, help="Path to the directory containing precomputed artifacts.")
    parser.add_argument("--out", type=str, default="submission.csv", help="Path for the output CSV file.")
    args = parser.parse_args()

    start_time = time.time()

    print("=" * 60)
    print("RANK.PY — Intelligent Candidate Discovery Engine")
    print("=" * 60)
    
    DEFAULT_JD = """We are seeking a Machine Learning Engineer with strong experience in search, ranking, and retrieval systems.
The ideal candidate will have expertise in vector databases (FAISS, Qdrant, Pinecone), dense embeddings, and learning-to-rank algorithms (LambdaMART). 
You should have strong Python and PyTorch skills, and experience deploying ML models to production APIs.
Familiarity with offline evaluation metrics like NDCG and MRR is highly desired."""

    cand_path = Path(args.candidates)
    art_dir = Path(args.artifacts)
    out_path = Path(args.out)

    if not cand_path.exists():
        print(f"ERROR: Candidates file not found at {cand_path}")
        sys.exit(1)
    if not art_dir.is_dir():
        print(f"ERROR: Artifacts directory not found at {art_dir}")
        sys.exit(1)

    # ---------------------------------------------------------------
    # Stage 2.1: Load Precomputed Artifacts
    # ---------------------------------------------------------------
    print("\n[1/5] Loading precomputed artifacts...")

    bm25_path = art_dir / 'bm25_index.pkl'
    try:
        with open(bm25_path, 'rb') as f:
            bm25 = pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: {bm25_path.name} not found in {art_dir}! Run precompute.py first.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load {bm25_path.name}. Is it corrupted? Details: {e}")
        sys.exit(1)

    meta_path = art_dir / 'candidate_metadata.pkl'
    try:
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: {meta_path.name} not found in {art_dir}! Run precompute.py first.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load {meta_path.name}. Is it corrupted? Details: {e}")
        sys.exit(1)

    candidate_ids = metadata['candidate_ids']
    career_texts = metadata['career_texts']
    m_cons = metadata['m_cons']
    m_traj = metadata['m_traj']
    m_behav = metadata['m_behav']
    s_skill = metadata['s_skill']
    m_loc = metadata['m_loc']
    reasoning_facts = metadata['reasoning_facts']
    stored_hash = metadata.get('dataset_hash')

    print("  Validating candidate dataset compatibility...")
    current_hash = compute_sha256(cand_path)
    if not stored_hash:
        print("  WARNING: Precomputed metadata is missing the dataset hash. Proceeding with caution.")
    elif current_hash != stored_hash:
        print(f"ERROR: Dataset hash mismatch!")
        print(f"  Supplied file: {current_hash}")
        print(f"  Artifact hash: {stored_hash}")
        print("  The candidates file has changed since precomputation. Please re-run precompute.py.")
        sys.exit(1)
    else:
        print("  Dataset verified successfully.")

    N = len(candidate_ids)
    skill_texts = metadata.get('skill_texts', [''] * N)
    print(f"  Loaded {N} candidates with BM25 index.")
    print(f"  Time: {time.time() - start_time:.1f}s")

    # ---------------------------------------------------------------
    # Stage 2.2: BM25 Sieve (100K → 10K)
    # ---------------------------------------------------------------
    print("\n[2/5] Running BM25 sieve against all 100K candidates...")

    if args.jd and Path(args.jd).exists():
        try:
            with open(Path(args.jd), 'r', encoding='utf-8') as f:
                jd_text = f.read().strip()
                if not jd_text:
                    print("ERROR: Job description file is empty.")
                    sys.exit(1)
        except Exception as e:
            print(f"ERROR: Failed to read job description from {args.jd}. Details: {e}")
            sys.exit(1)
    else:
        print("  No --jd file provided (or file not found). Using default embedded Job Description.")
        jd_text = DEFAULT_JD

    retrieval_query = build_retrieval_query(jd_text)
    jd_tokens = retrieval_query.lower().split()
    bm25_scores = np.array(bm25.get_scores(jd_tokens), dtype=np.float32)
    similarities = bm25_scores

    top_10k_indices = np.argsort(-bm25_scores)[:10000]

    print(f"  BM25 sieve complete.")
    print(f"  Score range: {bm25_scores[top_10k_indices[0]]:.4f} to {bm25_scores[top_10k_indices[-1]]:.4f}")
    print(f"  Time: {time.time() - start_time:.1f}s")

    # ---------------------------------------------------------------
    # Stage 2.3: Hadamard Product Gating
    # ---------------------------------------------------------------
    print("\n[3/5] Applying mathematical gating multipliers...")

    idx_arr = top_10k_indices
    sim_arr = similarities[idx_arr]
    gate_arr = m_cons[idx_arr] * m_traj[idx_arr] * m_behav[idx_arr] * m_loc[idx_arr]
    skill_arr = s_skill[idx_arr]

    gated_scores = sim_arr * gate_arr + 0.15 * skill_arr

    sorted_within = np.argsort(gated_scores)[::-1][:500]
    top_500_indices = idx_arr[sorted_within]

    dropped_honeypots = int(np.sum(m_cons[idx_arr] < 0.5))
    dropped_weak_traj = int(np.sum(m_traj[idx_arr] < 0.7))
    print(f"  Honeypots suppressed: {dropped_honeypots}")
    print(f"  Weak trajectories suppressed: {dropped_weak_traj}")
    print(f"  Top-500 candidates selected for deep reranking.")
    print(f"  Time: {time.time() - start_time:.1f}s")

    # ---------------------------------------------------------------
    # Stage 2.4: Deep Semantic Reranking (Cross-Encoder)
    # ---------------------------------------------------------------
    print("\n[4/5] Deep semantic reranking with Cross-Encoder (top 500)...")

    ce_model_path = art_dir / 'cross_encoder_model'
    if ce_model_path.exists():
        try:
            cross_encoder = CrossEncoder(str(ce_model_path), local_files_only=True)
            print("  Loaded cross-encoder from local cache.")
        except Exception as e:
            print(f"ERROR: Failed to load cross-encoder model. Details: {e}")
            sys.exit(1)
    else:
        print(f"ERROR: Local cross-encoder cache not found at {ce_model_path}.")
        print("Please ensure precompute.py completed successfully.")
        sys.exit(1)

    pairs = []
    for idx in top_500_indices:
        c_career = career_texts[idx]
        c_skills = skill_texts[idx]
        if len(c_career) > 400:
            career_summary = c_career[:200] + " ... " + c_career[-200:]
        else:
            career_summary = c_career
        structured = f"Skills: {c_skills[:100]}. Career: {career_summary[:300]}"
        pairs.append((jd_text, structured[:512]))

    ce_scores = cross_encoder.predict(pairs, show_progress_bar=True, batch_size=32)

    ce_arr = np.array(ce_scores, dtype=np.float32)
    ce_min, ce_max = ce_arr.min(), ce_arr.max()
    ce_norm = (ce_arr - ce_min) / (ce_max - ce_min + 1e-8)

    gate_arr_500 = (
        m_cons[top_500_indices] *
        m_traj[top_500_indices] *
        m_behav[top_500_indices] *
        m_loc[top_500_indices]
    )
    sim_arr_500 = similarities[top_500_indices] * gate_arr_500
    sim_min, sim_max = sim_arr_500.min(), sim_arr_500.max()
    sim_norm = (sim_arr_500 - sim_min) / (sim_max - sim_min + 1e-8)

    skill_arr_500 = s_skill[top_500_indices]

    final_scores = (
        0.60 * ce_norm +
        0.30 * sim_norm +
        0.10 * skill_arr_500
    ).astype(np.float32)

    print(f"  Cross-encoder reranking complete.")
    print(f"  Time: {time.time() - start_time:.1f}s")

    # ---------------------------------------------------------------
    # Stage 2.5: Output Generation
    # ---------------------------------------------------------------
    print("\n[5/5] Generating final submission...")

    top_100_within = np.argsort(final_scores)[::-1][:100]
    top_100_indices = top_500_indices[top_100_within]
    top_100_final = final_scores[top_100_within]
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])

            for rank, (idx, score) in enumerate(zip(top_100_indices, top_100_final), 1):
                cid = candidate_ids[idx]
                facts = reasoning_facts[idx]
                
                h = abs(hash(cid))
                title = facts['title']
                yoe = facts['yoe']
                
                str_list = facts['strengths']
                strengths = "; ".join(str_list) if str_list else "general engineering"
                
                concern_or_skill = ""
                if rank > 30:
                    c_sig = facts['concern_signal']
                    if c_sig:
                        has_word = any(w in c_sig.lower() for w in ['limited', 'gap', 'however', 'although', 'partial', 'concern'])
                        if has_word:
                            concern_or_skill = c_sig
                        else:
                            concern_or_skill = f"However, {c_sig}"
                    else:
                        concern_or_skill = "However, partial match due to general alignment gap."
                else:
                    s_part = facts['skill_part']
                    if s_part:
                        concern_or_skill = s_part
                        
                parts = []
                templates = [
                    f"Profile: {title} with {yoe}y experience. Notable strengths: {strengths}.",
                    f"Candidate background features {strengths}. Title is {title} ({yoe} years).",
                    f"Reviewing this record: {title}, {yoe}y history. Key expertise includes {strengths}.",
                    f"A {yoe}y {title} demonstrating {strengths}.",
                    f"Evaluated {title} ({yoe} years). Core competencies: {strengths}.",
                    f"Experience level: {yoe}y as {title}. Primary signals: {strengths}.",
                    f"{title} record with {yoe} years. Strong indicators in {strengths}.",
                    f"Assessing {title} at {yoe}y tenure. Detected {strengths}.",
                    f"This {title} brings {yoe} years of background. Focuses on {strengths}.",
                    f"Found {title} ({yoe}y). Evidence of {strengths} is present.",
                    f"Overview: {yoe}y of work as {title}. Highlights comprise {strengths}.",
                    f"Record analysis for {title} ({yoe}y). Main capabilities: {strengths}.",
                    f"Candidate possesses {yoe} years as {title}. Demonstrated abilities: {strengths}.",
                    f"Checking {title} profile. Tenure is {yoe}y, featuring {strengths}.",
                    f"Solid {title} with a {yoe}-year track record emphasizing {strengths}.",
                    f"History of {yoe} years. The {title} shows proficiency in {strengths}.",
                    f"The candidate is a {title} with {yoe}y experience. Key domain: {strengths}.",
                    f"Professional background: {title} ({yoe}y). Noteworthy for {strengths}.",
                    f"Inspecting {yoe}y {title} portfolio. Major strengths revolve around {strengths}.",
                    f"{title} evaluation ({yoe} years). Identified expertise: {strengths}.",
                    f"Details for {title}: {yoe} years active. Validated {strengths}.",
                    f"A {yoe}-year {title}. Career highlights showcase {strengths}.",
                    f"Profile assessment: {title}, {yoe}y. Primary focus lies in {strengths}.",
                    f"Review of {yoe}y {title} indicates core experience with {strengths}."
                ]
                
                parts.append(templates[h % 24])
                
                if facts['traj_part']:
                    t_part = facts['traj_part'].strip()
                    parts.append(t_part + ("" if t_part.endswith('.') else "."))
                    
                if concern_or_skill:
                    c_part = concern_or_skill.strip()
                    parts.append(c_part + ("" if c_part.endswith('.') else "."))
                    
                base = " ".join(parts)
                writer.writerow([cid, rank, round(float(score), 4), base])
    except Exception as e:
        print(f"ERROR: Failed to write to {out_path}. Details: {e}")
        sys.exit(1)

    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print("RANKING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Output file:  {out_path.name}")
    print(f"  Top 100 candidates ranked and saved.")
    print(f"  #1:   {candidate_ids[top_100_indices[0]]}  (score: {top_100_final[0]:.4f})")
    print(f"  #100: {candidate_ids[top_100_indices[-1]]}  (score: {top_100_final[-1]:.4f})")
    print(f"  Total time: {elapsed:.1f}s")
    if elapsed < 300:
        print(f"  ✅ PASSED: Under 5-minute limit ({elapsed:.1f}s / 300s)")
    else:
        print(f"  ❌ OVER LIMIT: {elapsed:.1f}s / 300s — optimize needed")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
