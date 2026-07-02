import os
import json
import gzip
import pickle
import re
import math
import sys
import subprocess
import time
from datetime import datetime, date
from collections import defaultdict

def install_and_import(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])

# Auto-install dependencies
install_and_import("numpy")
install_and_import("rank-bm25", "rank_bm25")
install_and_import("sentence-transformers", "sentence_transformers")

import numpy as np
from rank_bm25 import BM25Okapi


def main():
    start_time = time.time()

    print("=" * 60)
    print("PRECOMPUTE — Intelligent Candidate Discovery Engine")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Step 1.0: Load Dataset
    # ---------------------------------------------------------------
    print("\n[1/6] Loading candidates.jsonl.gz...")
    candidates = []
    try:
        with gzip.open("candidates.jsonl.gz", "rt") as f:
            for line in f:
                candidates.append(json.loads(line))
    except FileNotFoundError:
        print("ERROR: candidates.jsonl.gz not found!")
        return

    N = len(candidates)
    print(f"  Loaded {N} candidates. ({time.time() - start_time:.1f}s)")
    if candidates:
        print(f"  Sample candidate keys: {list(candidates[0].keys())}")
        ch = candidates[0].get('career_history', [])
        if ch:
            print(f"  Sample career_history[0] keys: {list(ch[0].keys())}")

    # ---------------------------------------------------------------
    # Step 1.1: Structural Chunking
    # ---------------------------------------------------------------
    print("\n[2/6] Building career text for each candidate...")

    candidate_ids = []
    career_texts = []
    skill_texts = []

    for i, c in enumerate(candidates):
        # --- Extract candidate ID (safely) ---
        cid = str(c.get('candidate_id', c.get('id', f'CAND_{i}')))
        candidate_ids.append(cid)

        # --- Build career text from career_history ---
        chunks = []
        for role in c.get('career_history', []):
            title = str(role.get('title') or 'Unknown')
            company = str(role.get('company') or 'Unknown')
            description = str(role.get('description') or '')
            chunks.append(f"{title} at {company}: {description}")

        career_text = ' '.join(chunks)

        # Fallback to career_text field if career_history was empty
        if not career_text.strip():
            career_text = str(c.get('career_text') or '')
        if not career_text.strip():
            career_text = str(c.get('title') or 'Unknown')

        # Extract skills separately — do NOT add to career_text used for BM25
        # (adding skills to BM25 text lets keyword stuffers rank highly via
        # retrieval even when their career descriptions contain nothing relevant)
        skills = c.get('skills', [])
        skill_names = []
        for s in skills:
            if isinstance(s, dict) and "name" in s:
                skill_names.append(str(s["name"]))
            elif isinstance(s, str):
                skill_names.append(s)
        skill_text = " ".join(skill_names)

        career_texts.append(career_text)
        skill_texts.append(skill_text)

    print(f"  Extracted career text for {len(career_texts)} candidates. ({time.time() - start_time:.1f}s)")

    # ---------------------------------------------------------------
    # Step 1.2: BM25 Sparse Index
    # ---------------------------------------------------------------
    print("\n[3/6] Building BM25 index...")
    print("  Tokenizing career texts...")

    # Tokenize career descriptions only — skill names are stored
    # separately and validated via s_skill, not included here.
    # Including skill names in BM25 lets keyword stuffers rank highly
    # via retrieval even with no career evidence.
    tokenized_corpus = []
    for text in career_texts:
        tokens = text.lower().split()
        tokenized_corpus.append(tokens if tokens else ["empty"])

    bm25 = BM25Okapi(tokenized_corpus)

    with open('bm25_index.pkl', 'wb') as f:
        pickle.dump(bm25, f)

    print(f"  BM25 index built over {N} candidates.")
    print(f"  Saved bm25_index.pkl (~40 MB, no 500MB embeddings file needed).")
    print(f"  BM25 build complete. ({time.time() - start_time:.1f}s)")

    # ---------------------------------------------------------------
    # Step 1.3: Pre-download Cross-Encoder for Offline Ranking
    # ---------------------------------------------------------------
    print("\n[4/6] Pre-downloading cross-encoder model for offline use...")
    from sentence_transformers import CrossEncoder
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    cross_encoder.save('cross_encoder_model')
    print(f"  Cross-encoder saved to cross_encoder_model/. ({time.time() - start_time:.1f}s)")

    # ---------------------------------------------------------------
    # Step 1.4: Mathematical Gating Multipliers
    # ---------------------------------------------------------------
    print("\n[5/6] Computing mathematical gating multipliers for all candidates...")

    today = date.today()

    m_cons = np.ones(N, dtype=np.float32)    # Internal consistency
    m_traj = np.ones(N, dtype=np.float32)    # Career trajectory
    m_behav = np.ones(N, dtype=np.float32)   # Behavioral signals
    s_skill = np.zeros(N, dtype=np.float32)  # Skill validation
    m_loc = np.ones(N, dtype=np.float32)     # Location + notice period

    reasoning_parts = [[] for _ in range(N)]

    # Only pure retrieval/vector AI skills used for honeypot detection.
    # CV/speech/deep learning are legitimate but unwanted — they are
    # penalized via m_traj, not flagged as honeypots here.
    # Including CV here would wrongly flag a genuine CV engineer as a
    # honeypot and eliminate them before gating even runs.
    ai_skills = {'faiss', 'embeddings', 'qdrant', 'pinecone', 'weaviate',
                 'lambdamart', 'vector database', 'vector search',
                 'semantic search', 'dense retrieval', 'llm', 'rag',
                 'bert', 'transformer', 'sentence-transformers'}
                 
    import re
    eng_keywords = {'engineer', 'scientist', 'developer', 'architect',
                    'researcher', 'ml', 'ai', 'data', 'machine learning',
                    'software', 'sde', 'swe'}
    eng_pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, eng_keywords)) + r')\b')
    
    relevant_domains = ['machine learning', 'artificial intelligence', 'data science',
                        'search', 'ranking', 'recommendation', 'embeddings', 'vector',
                        'retrieval', 'nlp', 'neural', 'deep learning', 'python',
                        'production', 'deployment', 'api', 'infrastructure',
                        'software engineer', 'backend', 'distributed', 'faiss',
                        'qdrant', 'elasticsearch', 'solr', 'lucene']
    domain_pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, relevant_domains)) + r')\b')
    
    mgmt_keywords = ['manager', 'director', 'vp', 'head of', 'chief']
    mgmt_pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, mgmt_keywords)) + r')\b')
    
    consulting_keywords = ['consultant', 'consulting', 'advisory', 'outsourcing']
    consulting_pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, consulting_keywords)) + r')\b')
    
    ml_phrases = ['machine learning', 'ai', 'vector', 'retrieval', 'data science', 'deep learning']
    ml_pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, ml_phrases)) + r')\b')

    for i, c in enumerate(candidates):
        career_history = c.get('career_history', [])

        # ===== M_cons: Internal Consistency =====
        # 1) Years of experience sanity check
        claimed_years = (
            c.get('total_years_experience') or
            c.get('years_of_experience') or
            c.get('total_experience_years') or
            c.get('experience_years') or
            c.get('yoe') or
            0
        )
        if not claimed_years:
            claimed_years = sum(
                float(r.get('duration_months') or 0)
                for r in c.get('career_history', [])
            ) / 12.0
        claimed_years = float(claimed_years)
        if claimed_years > 50:
            m_cons[i] *= 0.3
            reasoning_parts[i].append(f"Implausible {claimed_years}y experience claimed")

        # 2) Signup date sanity check (future dates)
        signup_str = str(c.get('signup_date') or '')
        try:
            signup_date = datetime.strptime(signup_str, '%Y-%m-%d').date()
            if (today - signup_date).days < 0:
                m_cons[i] *= 0.5
                reasoning_parts[i].append("Future signup date")
        except (ValueError, TypeError):
            pass

        # 3) Salary bounds check
        min_sal = c.get('expected_salary_min', 0) or 0
        max_sal = c.get('expected_salary_max', 0) or 0
        if min_sal > 0 and max_sal > 0 and min_sal > max_sal:
            m_cons[i] *= 0.5
            reasoning_parts[i].append("Salary min > max")

        # 4) Honeypot detection: fancy AI skills but no relevant job titles
        skills = c.get('skills', [])
        skill_names_lower = set()
        for s in skills:
            if isinstance(s, dict) and "name" in s:
                skill_names_lower.add(str(s["name"]).lower())
            elif isinstance(s, str):
                skill_names_lower.add(s.lower())

        titles_text = ' '.join(str(r.get('title') or '').lower() for r in career_history)
        has_ai_skills = bool(skill_names_lower & ai_skills)
        has_eng_title = bool(eng_pattern.search(titles_text))

        if has_ai_skills and not has_eng_title and len(career_history) > 0:
            m_cons[i] *= 0.4
            reasoning_parts[i].append("AI skills but no relevant titles (honeypot)")

        # ===== M_traj: Career Trajectory =====
        desc_combined = ' '.join(str(r.get('description') or '') for r in career_history).lower()
        domain_hits = len(set(domain_pattern.findall(desc_combined + " " + titles_text)))

        if domain_hits >= 5:
            m_traj[i] = 1.10
            reasoning_parts[i].append(f"Strong trajectory ({domain_hits} domain signals)")
        elif domain_hits >= 2:
            m_traj[i] = 1.00
        elif domain_hits >= 1:
            m_traj[i] = 0.85
            reasoning_parts[i].append("Weak trajectory relevance")
        else:
            m_traj[i] = 0.60
            reasoning_parts[i].append("No relevant career trajectory")

        # Pure management penalty
        is_pure_mgmt = bool(mgmt_pattern.search(titles_text)) and not has_eng_title
        if is_pure_mgmt:
            m_traj[i] *= 0.75
            reasoning_parts[i].append("Pure management (no engineering)")

        # Consulting lifer penalty
        consult_count = sum(
            1 for r in career_history
            if consulting_pattern.search(str(r.get('title') or '').lower())
        )
        if consult_count >= 3:
            m_traj[i] *= 0.70
            reasoning_parts[i].append("IT consulting lifer")

        # ===== M_behav: Behavioral =====
        reply_rate = c.get('recruiter_response_rate', 0.5)
        if reply_rate is None:
            reply_rate = 0.5

        last_active_str = str(c.get('last_active_date') or '')
        try:
            last_active_date = datetime.strptime(last_active_str, '%Y-%m-%d').date()
            inactive_days = (today - last_active_date).days
        except (ValueError, TypeError):
            inactive_days = 90

        activity_score = min(1.0, math.exp(-inactive_days / 90))

        interview_rate = c.get('interview_completion_rate', 0.5)
        if interview_rate is None:
            interview_rate = 0.5
        offer_rate = c.get('offer_acceptance_rate', 0.5)
        if offer_rate is None:
            offer_rate = 0.5

        m_behav[i] = round(0.85 + 0.15 * (
            0.30 * activity_score +
            0.25 * float(reply_rate) +
            0.25 * float(interview_rate) +
            0.20 * float(offer_rate)
        ), 3)

        if m_behav[i] >= 0.95:
            reasoning_parts[i].append("Highly responsive candidate")
        elif m_behav[i] < 0.90:
            reasoning_parts[i].append("Low engagement signals")

        # ===== S_skill: Skill Validation =====
        # Validate against career history descriptions ONLY — not career_texts[i]
        # which may contain appended skill names, making every skill trivially
        # validate and defeating the honeypot check entirely.
        desc_only = ' '.join(
            str(r.get('description') or '') for r in career_history
        ).lower()
        career_text_lower = desc_only
        validated = 0
        total = len(skill_names_lower)

        for skill in skill_names_lower:
            if skill in career_text_lower:
                # Exact match — full credit
                validated += 1
            else:
                words = [w for w in skill.split() if len(w) > 3]
                if not words:
                    continue
                if len(words) == 1:
                    # Single meaningful word — check if it appears
                    if words[0] in career_text_lower:
                        validated += 1
                else:
                    # Multi-word skill — count as validated if majority
                    # of meaningful tokens appear (handles terminology
                    # variation e.g. "vector search" vs "embedding search")
                    matches = sum(1 for w in words if w in career_text_lower)
                    if matches >= len(words) * 0.6:
                        validated += 1

        if total > 0:
            s_skill[i] = round((validated * 1.0 + (total - validated) * 0.30) / total, 3)
        else:
            s_skill[i] = 0.0

        # ML compound phrases check
        if not ml_pattern.search(career_text_lower):
            s_skill[i] = 0.05

        if s_skill[i] >= 0.5:
            reasoning_parts[i].append(f"Skills verified ({validated}/{total})")
        elif validated == 0 and total > 5:
            reasoning_parts[i].append(f"No skills evidenced in career text ({total} listed)")
        elif s_skill[i] < 0.2 and total > 5:
            reasoning_parts[i].append(f"Limited skill evidence ({validated}/{total} validated)")

        # ===== M_loc: Location & Notice Period =====
        location = str(c.get('location') or '').lower()
        notice = c.get('notice_period_days', 60)
        if notice is None:
            notice = 60

        if any(x in location for x in ['pune', 'noida', 'delhi', 'gurgaon', 'ncr']):
            loc_score = 1.05
        elif any(x in location for x in ['hyderabad', 'mumbai', 'bangalore', 'bengaluru']):
            loc_score = 1.00
        elif 'india' in location:
            loc_score = 0.90
        elif any(x in location for x in ['remote', 'anywhere']):
            loc_score = 0.95
        else:
            loc_score = 0.80

        if notice <= 30:
            notice_score = 1.00
        elif notice <= 60:
            notice_score = 0.95
        elif notice <= 90:
            notice_score = 0.85
        else:
            notice_score = 0.75

        m_loc[i] = round(loc_score * notice_score, 3)

        # Progress
        if (i + 1) % 25000 == 0:
            print(f"  Processed {i + 1}/{N} candidates... ({time.time() - start_time:.1f}s)")

    print(f"  All multipliers computed. ({time.time() - start_time:.1f}s)")

    # Build reasoning facts
    reasoning_facts = []
    for i, c in enumerate(candidates):
        parts = reasoning_parts[i]
        career_history_i = candidates[i].get('career_history', [])
        title = (
            c.get('title') or
            c.get('current_title') or
            c.get('current_role') or
            (career_history_i[0].get('title') if career_history_i else None) or
            'Unknown'
        )
        title = str(title)
        years = (
            c.get('total_years_experience') or
            c.get('years_of_experience') or
            c.get('total_experience_years') or
            c.get('experience_years') or
            c.get('yoe') or
            0
        )
        if not years:
            # Compute from career history durations as final fallback
            years = sum(
                float(r.get('duration_months') or 0)
                for r in c.get('career_history', [])
            ) / 12.0
        years = round(float(years), 1)

        # Build varied, specific reasoning using granular signals.
        # Each signal category picks the MOST SPECIFIC matching evidence
        # found in the candidate's actual career text, so different
        # candidates with different career histories get different phrases
        # even when they match the same broad category.

        desc_combined_r = ' '.join(
            str(r.get('description') or '') for r in c.get('career_history', [])
        ).lower()
        titles_text_r = ' '.join(
            str(r.get('title') or '') for r in c.get('career_history', [])
        ).lower()
        all_text_r = desc_combined_r + ' ' + titles_text_r

        # --- Signal A: Retrieval/Search specificity ---
        # Pick the most specific tool or method mentioned
        retrieval_signal = None
        if 'lambdamart' in all_text_r:
            retrieval_signal = "LambdaMART/LTR experience"
        elif 'bm25' in all_text_r:
            retrieval_signal = "BM25 hybrid search experience"
        elif 'elasticsearch' in all_text_r or 'solr' in all_text_r:
            retrieval_signal = "Elasticsearch/Solr search background"
        elif 'recommendation' in all_text_r:
            retrieval_signal = "recommendation systems background"
        elif 'ranking' in all_text_r and 'search' in all_text_r:
            retrieval_signal = "search ranking background"
        elif 'retrieval' in all_text_r or 'search' in all_text_r:
            retrieval_signal = "information retrieval background"

        # --- Signal B: Vector DB specificity ---
        vector_signal = None
        if 'faiss' in all_text_r:
            vector_signal = "FAISS vector index experience"
        elif 'qdrant' in all_text_r:
            vector_signal = "Qdrant vector DB experience"
        elif 'pinecone' in all_text_r:
            vector_signal = "Pinecone vector DB experience"
        elif 'weaviate' in all_text_r:
            vector_signal = "Weaviate vector DB experience"
        elif 'milvus' in all_text_r:
            vector_signal = "Milvus vector DB experience"
        elif 'vector' in all_text_r and 'embedding' in all_text_r:
            vector_signal = "vector embedding pipeline experience"
        elif 'embedding' in all_text_r:
            vector_signal = "embedding-based retrieval experience"
        elif 'vector' in all_text_r:
            vector_signal = "vector search experience"

        # --- Signal C: Evaluation specificity ---
        eval_signal = None
        if 'ndcg' in all_text_r and 'mrr' in all_text_r:
            eval_signal = "NDCG + MRR evaluation expertise"
        elif 'ndcg' in all_text_r:
            eval_signal = "NDCG ranking evaluation experience"
        elif 'mrr' in all_text_r or 'map' in all_text_r:
            eval_signal = "MRR/MAP evaluation experience"
        elif 'a/b test' in all_text_r or 'ab test' in all_text_r:
            eval_signal = "A/B testing and offline evaluation"
        elif 'offline eval' in all_text_r or 'evaluation' in all_text_r:
            eval_signal = "offline evaluation framework experience"

        # --- Signal D: Production specificity ---
        prod_signal = None
        if 'shipped' in all_text_r and ('million' in all_text_r
                                         or 'billion' in all_text_r):
            prod_signal = "shipped ML at scale (millions of users)"
        elif 'latency' in all_text_r and 'production' in all_text_r:
            prod_signal = "production ML with latency optimization"
        elif 'microservice' in all_text_r or 'api' in all_text_r:
            prod_signal = "production ML API/microservice deployment"
        elif 'deployed' in all_text_r or 'shipped' in all_text_r:
            prod_signal = "production ML deployment"
        elif 'production' in all_text_r:
            prod_signal = "production ML environment experience"

        # --- Signal E: Python/tooling specificity ---
        python_signal = None
        if 'pytorch' in all_text_r and 'python' in all_text_r:
            python_signal = "Python + PyTorch ML stack"
        elif 'tensorflow' in all_text_r and 'python' in all_text_r:
            python_signal = "Python + TensorFlow ML stack"
        elif 'python' in all_text_r and 'spark' in all_text_r:
            python_signal = "Python + Spark data engineering"
        elif 'python' in all_text_r:
            python_signal = "Python evidenced in career descriptions"

        # Gap suffix — complete sentences only, based on which JD signals
        # are actually missing for this specific candidate.
        gap_suffix = None
        if retrieval_signal is None and vector_signal is None:
            gap_suffix = "Gap: limited retrieval or vector search evidence in career text"
        elif eval_signal is None and prod_signal is None:
            gap_suffix = "Gap: no evaluation framework or production deployment evidence"
        elif retrieval_signal is None:
            gap_suffix = "Gap: limited direct retrieval system experience evidenced"
        elif vector_signal is None:
            gap_suffix = "Gap: no vector database experience evidenced in career text"
        elif eval_signal is None:
            gap_suffix = "Gap: evaluation framework experience (NDCG/MAP) not evidenced"
        elif prod_signal is None:
            gap_suffix = "Gap: production ML deployment not evidenced in career text"
        else:
            gap_suffix = "Note: strong signals but lower overall match confidence"

        # --- Assemble reasoning facts ---
        # Pick up to 3 positive signals, prioritizing the most specific
        positive_signals = [s for s in [
            retrieval_signal, vector_signal, eval_signal,
            prod_signal, python_signal
        ] if s is not None]

        fact_dict = {
            'title': title,
            'yoe': years,
            'strengths': positive_signals[:2],
            'traj_part': next((p for p in parts if 'trajectory' in p), None),
            'concern_signal': gap_suffix,
            'skill_part': next((p for p in parts if 'skill' in p.lower() or 'Skills' in p), None)
        }
        reasoning_facts.append(fact_dict)

    # ---------------------------------------------------------------
    # Step 1.5: Save All Artifacts
    # ---------------------------------------------------------------
    print("\n[6/6] Saving all artifacts...")

    metadata = {
        'candidate_ids': candidate_ids,
        'career_texts': career_texts,
        'skill_texts': skill_texts,
        'm_cons': m_cons,
        'm_traj': m_traj,
        'm_behav': m_behav,
        's_skill': s_skill,
        'm_loc': m_loc,
        'reasoning_facts': reasoning_facts,
    }

    with open('candidate_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    # Ground truth canary checks — diagnostic only, does not block saving
    if 'CAND_0000031' in candidate_ids:
        idx31 = candidate_ids.index('CAND_0000031')
        print(f"\n[CANARY] CAND_0000031 found at index {idx31}")
        print(f"  m_cons={m_cons[idx31]:.3f}  m_traj={m_traj[idx31]:.3f}"
              f"  m_behav={m_behav[idx31]:.3f}  s_skill={s_skill[idx31]:.3f}"
              f"  m_loc={m_loc[idx31]:.3f}")
        print(f"  reasoning facts: {reasoning_facts[idx31]}")
    else:
        print("\n[CANARY] WARNING: CAND_0000031 not found in dataset")

    if 'CAND_0000021' in candidate_ids:
        idx21 = candidate_ids.index('CAND_0000021')
        print(f"[CANARY] CAND_0000021 at index {idx21}  "
              f"m_cons={m_cons[idx21]:.3f}  s_skill={s_skill[idx21]:.3f}")

    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print("PRECOMPUTE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total candidates:                {N}")
    print(f"  Honeypots detected (M_cons<0.5): {int(np.sum(m_cons < 0.5))}")
    print(f"  Strong trajectory (M_traj>=1.05):{int(np.sum(m_traj >= 1.05))}")
    print(f"  Weak trajectory (M_traj<0.7):    {int(np.sum(m_traj < 0.7))}")
    print(f"  High skill match (S_skill>=0.7): {int(np.sum(s_skill >= 0.7))}")
    print(f"  Total time:                      {elapsed:.1f}s")
    print(f"{'=' * 60}")
    print("Precompute complete! Run rank.py next.")


if __name__ == "__main__":
    main()
