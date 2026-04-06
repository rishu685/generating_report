#!/usr/bin/env python3
"""
Intelligence Scoring Engine

This module implements a 4-factor scoring system for evaluating internship candidates:
- Technical Skills (35%): Weighted by skill demand and mastery
- Answer Quality (30%): Length, detail, reasoning, originality
- GitHub Activity (20%): Repositories, contributions, stars, forks
- Profile Completeness (15%): Field coverage with penalties for missing data

Anti-Cheat: TF-IDF similarity detection, timing analysis, AI marker detection
"""

import json
import csv
import argparse
import random
import math
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict
import statistics


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Candidate:
    """Represents a candidate application."""
    candidate_id: str
    name: str
    technical_skills: Dict[str, int]  # {"python": 5, "sql": 4, ...}
    answers: List[str]  # List of answers to open-ended questions
    github: Dict[str, int]  # {"repos": 12, "contributions": 450, "stars": 89, "forks": 12}
    profile: Dict[str, str]  # {"email": "...", "linkedin": "...", ...}
    response_seconds: int  # Time taken to complete the assessment


@dataclass
class CandidateResult:
    """Scoring results for a single candidate."""
    candidate_id: str
    name: str
    technical_skills: Dict[str, int]
    answers: float  # Score 0-100
    github: float  # Score 0-100
    profile: float  # Score 0-100
    response_seconds: int
    total: float  # Total score 0-100
    tier: str  # "Fast-Track", "Standard", "Review", "Reject"
    technical: float  # Technical skills score 0-100
    originality_penalty: float  # 0-12 points
    empty_profile_penalty: float  # 0-18 points
    similarity_penalty: float  # 0-16 points
    timing_penalty: float  # 0-12 points
    flags: List[str] = field(default_factory=list)  # ["ai_like_text", "fast_completion", ...]
    reasons: List[str] = field(default_factory=list)  # Human-readable explanations
    rank: int = 0  # Rank in overall sorted list


# ============================================================================
# SKILL WEIGHTING
# ============================================================================

SKILL_WEIGHTS = {
    "python": 2.2,
    "javascript": 1.8,
    "sql": 1.8,
    "dsa": 1.8,
    "system_design": 2.0,
    "communication": 1.5,
    "ml": 2.3,
    "java": 1.6,
    "go": 1.9,
    "rust": 2.1,
}

AI_MARKERS = [
    "as an ai language model",
    "here is a concise",
    "here's a concise",
    "here are the steps",
    "i don't have personal",
    "as an artificial intelligence",
    "as a machine learning model",
]

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "to", "was", "will", "with", "i", "you", "we", "they",
    "this", "these", "those", "my", "your", "what", "which", "how",
}

TIER_THRESHOLDS = {
    "Fast-Track": 80,
    "Standard": 65,
    "Review": 48,
    "Reject": 0,
}


# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

def score_technical(technical_skills: Dict[str, int]) -> float:
    """
    Score technical skills as weighted average.
    
    Skills are weighted by demand (python 2.2x, ml 2.3x, etc.).
    Max skill level is 5, so max weighted score is calculated accordingly.
    """
    if not technical_skills:
        return 0.0
    
    total_weighted = sum(level * SKILL_WEIGHTS.get(skill, 1.0) 
                         for skill, level in technical_skills.items())
    max_weighted = sum(5 * SKILL_WEIGHTS.get(skill, 1.0) 
                       for skill in technical_skills.keys())
    
    if max_weighted == 0:
        return 0.0
    return min(100.0, (total_weighted / max_weighted) * 100)


def score_answers(answers: List[str]) -> float:
    """
    Score answer quality on multiple criteria:
    - Length (35 pts): Detailed answers are 400+ chars
    - Detail markers (10 pts): "example", "specifically", "particular"
    - Numbers/data (10 pts): Contains digits or quantification
    - Reasoning (20 pts): Contains "why", "because", "reason"
    - Originality (25 pts): Low AI marker score, diverse vocabulary
    """
    if not answers:
        return 0.0
    
    score = 0.0
    text = " ".join(answers).lower()
    
    # Length (35 pts)
    avg_length = statistics.mean(len(a) for a in answers)
    if avg_length >= 400:
        score += 35
    elif avg_length >= 200:
        score += 25
    else:
        score += max(0, int((avg_length / 200) * 25))
    
    # Detail markers (10 pts)
    detail_words = ["example", "specifically", "particular", "instance", "case"]
    if any(word in text for word in detail_words):
        score += 10
    
    # Numbers/data (10 pts)
    if any(c.isdigit() for c in text):
        score += 10
    
    # Reasoning (20 pts)
    reasoning_words = ["why", "because", "reason", "therefore", "thus"]
    if any(word in text for word in reasoning_words):
        score += 20
    
    # Originality (25 pts)
    unique_words = len(set(text.split()))
    total_words = len(text.split())
    if total_words > 0:
        diversity = unique_words / total_words
        if diversity > 0.5:
            score += 25
        else:
            score += int(diversity * 50)
    
    return min(100.0, score)


def score_github(github: Dict[str, int]) -> float:
    """
    Score GitHub activity:
    - repos * 7
    - contributions / 6
    - stars * 2
    - forks * 1.5
    Capped at 100.
    """
    if not github:
        return 0.0
    
    score = (
        github.get("repos", 0) * 7 +
        github.get("contributions", 0) / 6 +
        github.get("stars", 0) * 2 +
        github.get("forks", 0) * 1.5
    )
    return min(100.0, score)


def score_profile(profile: Dict[str, str]) -> float:
    """
    Score profile completeness.
    Expected fields: email, linkedin, portfolio, experience, education, phone
    Penalty: 18 points if fewer than 3 fields filled.
    """
    filled_fields = sum(1 for v in profile.values() if v and str(v).strip())
    score = 100.0
    
    if filled_fields < 3:
        score -= 18  # Penalty for incomplete profile
    
    return max(0.0, score)


def originality_penalty(answers: List[str]) -> float:
    """
    Detect AI-like text or low originality.
    - AI marker detection: 12 pts if detected
    - Lexical diversity check: 
    """
    if not answers:
        return 0.0
    
    text = " ".join(answers).lower()
    
    # Check for AI markers
    for marker in AI_MARKERS:
        if marker in text:
            return 12.0
    
    # Lexical diversity: too many repeated words = suspicious
    words = [w for w in text.split() if len(w) > 2]
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.2:  # Very low diversity
            return 8.0
    
    return 0.0


def timing_penalty(response_seconds: int) -> float:
    """
    Penalize suspiciously fast submissions (<180 seconds).
    """
    if response_seconds < 180:
        return 12.0
    return 0.0


def tokenize(text: str) -> List[str]:
    """Extract content words for TF-IDF (filter stop words, min 3 chars)."""
    words = text.lower().split()
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]


def build_tfidf_vectors(candidates: List[Candidate]) -> Dict[str, Dict[str, float]]:
    """
    Build TF-IDF vectors for each candidate's answers.
    Used for detecting copy-pasted/similar answers.
    """
    # Tokenize all answers
    all_docs = {}
    for c in candidates:
        text = " ".join(c.answers)
        all_docs[c.candidate_id] = tokenize(text)
    
    # Calculate IDF
    idf = {}
    total_docs = len(all_docs)
    word_doc_count = defaultdict(int)
    
    for doc in all_docs.values():
        for word in set(doc):
            word_doc_count[word] += 1
    
    for word, count in word_doc_count.items():
        idf[word] = math.log(total_docs / count) if count > 0 else 0
    
    # Calculate TF-IDF
    tfidf = {}
    for cand_id, doc in all_docs.items():
        if not doc:
            tfidf[cand_id] = {}
            continue
        
        term_freq = defaultdict(int)
        for word in doc:
            term_freq[word] += 1
        
        tfidf[cand_id] = {
            word: (count / len(doc)) * idf.get(word, 0)
            for word, count in term_freq.items()
        }
    
    return tfidf


def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """Calculate cosine similarity between two TF-IDF vectors."""
    dot_product = sum(vec1.get(w, 0) * vec2.get(w, 0) for w in set(vec1) | set(vec2))
    norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def union_groups(similarities: Dict[Tuple[str, str], float], threshold: float = 0.97) -> Dict[str, Set[str]]:
    """
    Find groups of similar candidates using union-find.
    Returns mapping of group_id -> set of candidate_ids.
    """
    parent = {}
    
    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Build similarity pairs above threshold
    for (c1, c2), sim in similarities.items():
        if sim >= threshold:
            union(c1, c2)
    
    # Group by parent
    groups = defaultdict(set)
    for cand_id in parent.keys():
        root = find(cand_id)
        groups[root].add(cand_id)
    
    return groups


def detect_similarity_penalties(candidates: List[Candidate]) -> Dict[str, Tuple[float, str]]:
    """
    Detect high-similarity answers (copy-pasting).
    Return dict: candidate_id -> (penalty_points, copy_ring_description)
    """
    # Build TF-IDF vectors
    tfidf = build_tfidf_vectors(candidates)
    
    # Compute all pairwise similarities
    similarities = {}
    for i, c1 in enumerate(candidates):
        for j, c2 in enumerate(candidates):
            if i < j:
                sim = cosine_similarity(tfidf.get(c1.candidate_id, {}), tfidf.get(c2.candidate_id, {}))
                if sim > 0.85:  # Only store notable similarities
                    similarities[(c1.candidate_id, c2.candidate_id)] = sim
    
    # Find copy rings (3+ identical)
    groups = union_groups(similarities, threshold=0.97)
    
    penalties = {}
    for group_id, group_members in groups.items():
        if len(group_members) >= 3:
            # Copy ring detected: assign penalty to all members
            for member_id in group_members:
                penalties[member_id] = (16.0, f"copy_ring with {len(group_members)-1} others")
    
    # Single high similarities get lower penalty
    for (c1, c2), sim in similarities.items():
        if sim >= 0.97 and c1 not in penalties and c2 not in penalties:
            penalties[c1] = (8.0, f"high_similarity_to_{c2}")
            penalties[c2] = (8.0, f"high_similarity_to_{c1}")
    
    return penalties


def evaluate_candidates(candidates: List[Candidate]) -> List[CandidateResult]:
    """Score all candidates and return results with ranks."""
    
    # Detect similarity penalties
    similarity_penalties = detect_similarity_penalties(candidates)
    
    results = []
    for rank, candidate in enumerate(candidates, 1):
        # Score each component
        tech_score = score_technical(candidate.technical_skills)
        answers_score = score_answers(candidate.answers)
        github_score = score_github(candidate.github)
        profile_score = score_profile(candidate.profile)
        
        # Get penalties
        orig_penalty = originality_penalty(candidate.answers)
        empty_profile_penalty = 18.0 if score_profile(candidate.profile) < 100 else 0.0
        sim_penalty, sim_reason = similarity_penalties.get(candidate.candidate_id, (0.0, ""))
        tim_penalty = timing_penalty(candidate.response_seconds)
        
        # Total score: weighted components - penalties
        total = (
            tech_score * 0.35 +
            answers_score * 0.30 +
            github_score * 0.20 +
            profile_score * 0.15
        ) - orig_penalty - empty_profile_penalty - sim_penalty - tim_penalty
        
        total = max(0.0, total)
        
        # Assign tier
        tier = "Reject"
        for t, threshold in sorted(TIER_THRESHOLDS.items(), key=lambda x: -x[1]):
            if total >= threshold:
                tier = t
                break
        
        # Collect flags and reasons
        flags = []
        reasons = []
        
        if orig_penalty > 0:
            flags.append("ai_like_text")
            reasons.append("AI-like language markers or low lexical variety")
        
        if empty_profile_penalty > 0:
            flags.append("empty_profile")
            reasons.append("Profile is missing core fields")
        
        if sim_penalty > 0:
            flags.append("copy_ring")
            reasons.append(sim_reason)
        
        if tim_penalty > 0:
            flags.append("fast_completion")
            reasons.append("Submission timing is unusually fast")
        
        if total < 65:
            flags.append("manual_review")
            reasons.append("Score is below the review threshold")
        
        result = CandidateResult(
            candidate_id=candidate.candidate_id,
            name=candidate.name,
            technical_skills=candidate.technical_skills,
            answers=answers_score,
            github=github_score,
            profile=profile_score,
            response_seconds=candidate.response_seconds,
            total=round(total, 2),
            tier=tier,
            technical=round(tech_score, 2),
            originality_penalty=orig_penalty,
            empty_profile_penalty=empty_profile_penalty,
            similarity_penalty=sim_penalty,
            timing_penalty=tim_penalty,
            flags=flags,
            reasons=reasons,
            rank=rank,
        )
        results.append(result)
    
    return results


# ============================================================================
# FILE I/O FUNCTIONS
# ============================================================================

def candidate_from_mapping(data: Dict[str, Any]) -> Candidate:
    """Convert dict to Candidate (flexible type coercion)."""
    try:
        technical_skills = data.get("technical_skills", {})
        if isinstance(technical_skills, dict):
            technical_skills = {k: int(v) for k, v in technical_skills.items()}
        
        github = data.get("github", {})
        if isinstance(github, dict):
            github = {k: int(v) for k, v in github.items()}
        
        answers = data.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]
        answers = [str(a) for a in answers]
        
        profile = data.get("profile", {})
        if isinstance(profile, dict):
            profile = {k: str(v) for k, v in profile.items()}
        
        return Candidate(
            candidate_id=str(data.get("candidate_id", "")),
            name=str(data.get("name", "")),
            technical_skills=technical_skills,
            answers=answers,
            github=github,
            profile=profile,
            response_seconds=int(data.get("response_seconds", 300)),
        )
    except Exception as e:
        raise ValueError(f"Failed to parse candidate: {e}")


def load_candidates_from_json(filepath: Path) -> List[Candidate]:
    """Load candidates from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
        if isinstance(data, list):
            return [candidate_from_mapping(item) for item in data]
        else:
            return [candidate_from_mapping(data)]


def load_candidates_from_csv(filepath: Path) -> List[Candidate]:
    """Load candidates from CSV file."""
    candidates = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidates.append(candidate_from_mapping(row))
    return candidates


def load_candidates(filepath: Path) -> List[Candidate]:
    """Load candidates from JSON, CSV, or JSONL."""
    if filepath.suffix == ".json":
        return load_candidates_from_json(filepath)
    elif filepath.suffix == ".csv":
        return load_candidates_from_csv(filepath)
    elif filepath.suffix == ".jsonl":
        candidates = []
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    candidates.append(candidate_from_mapping(data))
        return candidates
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def export_results(results: List[CandidateResult], output_json: Path = None, output_csv: Path = None):
    """Export results to JSON and/or CSV."""
    # Prepare export data
    export_data = []
    for r in results:
        export_data.append({
            "rank": r.rank,
            "candidate_id": r.candidate_id,
            "name": r.name,
            "technical_skills": r.technical_skills,
            "technical": r.technical,
            "answers": round(r.answers, 2),
            "github": round(r.github, 2),
            "profile": round(r.profile, 2),
            "total": r.total,
            "tier": r.tier,
            "originality_penalty": r.originality_penalty,
            "empty_profile_penalty": r.empty_profile_penalty,
            "similarity_penalty": r.similarity_penalty,
            "timing_penalty": r.timing_penalty,
            "flags": r.flags,
            "reasons": r.reasons,
        })
    
    # JSON export
    if output_json:
        with open(output_json, "w") as f:
            json.dump(export_data, f, indent=2)
        print(f"✅ Exported JSON: {output_json}")
    
    # CSV export
    if output_csv:
        with open(output_csv, "w", newline="") as f:
            fieldnames = [
                "rank", "candidate_id", "name", "tier",
                "technical", "answers", "github", "profile", "total",
                "originality_penalty", "empty_profile_penalty", "similarity_penalty", "timing_penalty",
                "flags", "reasons"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in export_data:
                writer.writerow({k: row.get(k) for k in fieldnames})
        print(f"✅ Exported CSV: {output_csv}")


def summarize_batch(results: List[CandidateResult], batch_size: int = 100) -> List[Dict[str, Any]]:
    """Summarize results by batch for learning insights."""
    batches = []
    for i in range(0, len(results), batch_size):
        batch = results[i:i+batch_size]
        avg_score = statistics.mean(r.total for r in batch)
        flag_counts = defaultdict(int)
        for r in batch:
            for flag in r.flags:
                flag_counts[flag] += 1
        
        top_tier = sum(1 for r in batch if r.tier == "Fast-Track")
        
        batches.append({
            "batch_size": len(batch),
            "average_total": round(avg_score, 2),
            "flag_counts": dict(flag_counts),
            "top_tier_share": round(top_tier / len(batch), 3) if batch else 0,
        })
    
    return batches


def summarize_cohort(results: List[CandidateResult]) -> Dict[str, Any]:
    """Summarize overall cohort statistics."""
    if not results:
        return {}
    
    tier_counts = defaultdict(int)
    flag_counts = defaultdict(int)
    
    for r in results:
        tier_counts[r.tier] += 1
        for flag in r.flags:
            flag_counts[flag] += 1
    
    return {
        "count": len(results),
        "tier_counts": dict(tier_counts),
        "average_score": round(statistics.mean(r.total for r in results), 2),
        "flag_counts": dict(flag_counts),
        "average_technical": round(statistics.mean(r.technical for r in results), 2),
        "average_answers": round(statistics.mean(r.answers for r in results), 2),
        "average_github": round(statistics.mean(r.github for r in results), 2),
    }


def generate_synthetic_candidates(count: int = 10) -> List[Candidate]:
    """Generate realistic synthetic candidates for testing."""
    answer_templates = [
        [
            "I broke the problem into layers, added logging, and measured each stage before changing the design.",
            "For example, I reduced rework by documenting the assumptions and comparing the outputs at every step.",
        ],
        [
            "I started by understanding the constraints and requirements from stakeholders.",
            "Then I designed a solution that could scale with future needs.",
        ],
        [
            "The key insight was to reuse existing patterns from the codebase.",
            "This reduced bugs and made the feature easier to maintain.",
        ],
        [
            "I built a prototype first to validate the approach with real data.",
            "After feedback, I refined it twice before the final version.",
        ],
        [
            "Performance was critical, so I profiled the code to find bottlenecks.",
            "Using caching and batch processing, I improved throughput by 3x.",
        ],
        [
            "I collaborated with the design team to ensure the API was intuitive.",
            "We iterated based on user feedback from beta testing.",
        ],
    ]
    
    candidates = []
    for i in range(count):
        cand_id = f"S{i+1:04d}"
        name = f"Candidate {i+1}"
        
        # Vary skills and profiles
        base_skill = random.randint(1, 5)
        tech_skills = {
            "python": max(1, base_skill + random.randint(-1, 1)),
            "sql": max(1, base_skill + random.randint(-1, 1)),
            "javascript": max(1, base_skill + random.randint(-2, 1)),
            "communication": max(1, base_skill + random.randint(-1, 2)),
            "system_design": max(1, base_skill + random.randint(-1, 1)),
            "dsa": max(1, base_skill + random.randint(-1, 1)),
            "ml": max(1, random.randint(0, 4)),
        }
        
        answers = random.choice(answer_templates)
        
        github = {
            "repos": max(0, base_skill * 2 + random.randint(-2, 5)),
            "contributions": max(0, base_skill * 100 + random.randint(-50, 200)),
            "stars": max(0, base_skill * 10 + random.randint(-5, 30)),
            "forks": max(0, base_skill * 3 + random.randint(-2, 8)),
        }
        
        # Some candidates missing profile fields
        profile_filled = random.random() > 0.2
        profile = {
            "email": "candidate@example.com" if profile_filled else "",
            "linkedin": "linkedin.com/in/candidate" if profile_filled else "",
            "portfolio": "portfolio.com" if profile_filled and random.random() > 0.5 else "",
            "experience": f"{base_skill-1} years" if profile_filled else "",
            "education": "University" if profile_filled else "",
            "phone": "+91-xxxx-xxxx" if profile_filled and random.random() > 0.7 else "",
        }
        
        response_time = random.randint(120, 900)  # Some suspiciously fast
        
        candidates.append(Candidate(
            candidate_id=cand_id,
            name=name,
            technical_skills=tech_skills,
            answers=answers,
            github=github,
            profile=profile,
            response_seconds=response_time,
        ))
    
    return candidates


# ============================================================================
# CLI & MAIN
# ============================================================================

def print_ranked_preview(results: List[CandidateResult], top_n: int = 10):
    """Print ranked results as formatted table."""
    print("\nRanked applicants\n")
    for r in results[:top_n]:
        print(f"#{r.rank:04d} {r.candidate_id} {r.name} | {r.tier} | "
              f"total={r.total} | "
              f"tech={r.technical} answers={r.answers} github={r.github} profile={r.profile} | "
              f"penalties={r.originality_penalty + r.empty_profile_penalty + r.similarity_penalty + r.timing_penalty} | "
              f"flags={','.join(r.flags) if r.flags else 'none'}")


def print_cohort_summary(summary: Dict[str, Any]):
    """Print cohort statistics."""
    print("\n\nCohort summary\n")
    print(json.dumps(summary, indent=2))


def print_batch_summaries(batches: List[Dict[str, Any]]):
    """Print batch learning summaries."""
    print("\n\nBatch learning summaries\n")
    print(json.dumps(batches, indent=2))


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Intelligent Candidate Screening & Ranking Engine")
    
    parser.add_argument("--input", type=Path, help="Input file (JSON/CSV/JSONL)")
    parser.add_argument("--generate", type=int, default=10, help="Generate N synthetic candidates")
    parser.add_argument("--top-n", type=int, default=10, help="Print top N results")
    parser.add_argument("--chunk-size", type=int, default=100, help="Batch size for learning summaries")
    parser.add_argument("--output-json", type=Path, help="Export results to JSON")
    parser.add_argument("--output-csv", type=Path, help="Export results to CSV")
    parser.add_argument("--no-preview", action="store_true", help="Skip result preview")
    parser.add_argument("--no-cohort", action="store_true", help="Skip cohort summary")
    parser.add_argument("--no-batch", action="store_true", help="Skip batch summaries")
    
    return parser.parse_args()


def load_input_candidates(args) -> List[Candidate]:
    """Load candidates from input file or generate synthetic."""
    if args.input:
        return load_candidates(args.input)
    else:
        return generate_synthetic_candidates(args.generate)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load candidates
    print(f"Loading candidates...")
    candidates = load_input_candidates(args)
    print(f"Loaded {len(candidates)} candidates\n")
    
    # Evaluate
    print(f"Evaluating candidates...")
    results = evaluate_candidates(candidates)
    
    # Sort by total score descending
    results.sort(key=lambda x: x.total, reverse=True)
    for i, r in enumerate(results, 1):
        r.rank = i
    
    print(f"Evaluation complete.\n")
    
    # Print preview
    if not args.no_preview:
        print_ranked_preview(results, args.top_n)
    
    # Print cohort summary
    if not args.no_cohort:
        summary = summarize_cohort(results)
        print_cohort_summary(summary)
    
    # Print batch summaries
    if not args.no_batch and len(results) > args.chunk_size:
        batches = summarize_batch(results, args.chunk_size)
        print_batch_summaries(batches)
    
    # Export
    if args.output_json or args.output_csv:
        export_results(results, args.output_json, args.output_csv)
    
    # Print JSON summary if exporting
    if args.output_json:
        export_list = []
        for r in results:
            export_list.append({
                "candidate_id": r.candidate_id,
                "name": r.name,
                "technical_skills": r.technical_skills,
                "answers": round(r.answers, 2),
                "github": round(r.github, 2),
                "profile": round(r.profile, 2),
                "response_seconds": r.response_seconds,
                "total": r.total,
                "tier": r.tier,
                "technical": r.technical,
                "originality_penalty": r.originality_penalty,
                "empty_profile_penalty": r.empty_profile_penalty,
                "similarity_penalty": r.similarity_penalty,
                "timing_penalty": r.timing_penalty,
                "flags": r.flags,
                "reasons": r.reasons,
                "rank": r.rank,
            })
        print("\n\nJSON summary\n")
        print(json.dumps(export_list, indent=2)[:2000])


if __name__ == "__main__":
    main()
