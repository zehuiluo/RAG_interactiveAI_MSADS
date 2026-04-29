"""
evaluate.py  —  Evaluation metrics for the MSADS RAG system.

Metrics
-------
1. Retrieval Precision@K   — what fraction of top-K results are relevant?
2. Mean Reciprocal Rank    — where does the first relevant result appear?
3. Answer Faithfulness     — does the LLM answer stay grounded in context?
4. Keyword Coverage        — do answers contain expected keywords?

Faithfulness (Method A — lightweight, no external API needed):
  For each sentence in the answer, check if any meaningful phrase (4+ chars)
  from that sentence appears in the retrieved context. Score = fraction of
  sentences that are grounded in context.
"""

import re
import json
from typing import List, Dict
from vector_store import MSADSVectorStore

# ── Gold-standard test set (20 questions) ─────────────────────────────────────
TEST_SET: List[Dict] = [
    {
        "question": "What are the core courses in the MS in Applied Data Science?",
        "expected_keywords": ["machine learning", "statistical", "data engineering",
                              "capstone", "python"],
        "relevant_source_titles": ["Core Courses", "Program Tracks and Structure"],
    },
    {
        "question": "What are the admission requirements for the MSADS program?",
        "expected_keywords": ["bachelor", "recommendation", "statement", "resume",
                              "toefl", "programming"],
        "relevant_source_titles": ["Admissions Requirements"],
    },
    {
        "question": "How much does the program cost?",
        "expected_keywords": ["tuition", "scholarship", "fees"],
        "relevant_source_titles": ["Tuition and Financial Aid"],
    },
    {
        "question": "What career outcomes do MSADS graduates achieve?",
        "expected_keywords": ["data scientist", "google", "amazon", "jpmorgan",
                              "engineer"],
        "relevant_source_titles": ["Career Outcomes"],
    },
    {
        "question": "Can I take the MSADS program part-time?",
        "expected_keywords": ["part-time", "quarter", "flexible"],
        "relevant_source_titles": ["In-Person Program Format", "Online Program"],
    },
    {
        "question": "Is the MSADS degree STEM OPT eligible?",
        "expected_keywords": ["stem", "opt", "visa"],
        "relevant_source_titles": ["Visa and International Students"],
    },
    {
        "question": "What is the capstone project in the MSADS program?",
        "expected_keywords": ["capstone", "industry", "real-world", "quarter"],
        "relevant_source_titles": ["Capstone Project", "Program Tracks and Structure"],
    },
    {
        "question": "What electives are available in the MSADS program?",
        "expected_keywords": ["generative ai", "nlp", "elective"],
        "relevant_source_titles": ["Elective Courses"],
    },
    {
        "question": "When is the application deadline for MSADS?",
        "expected_keywords": ["deadline", "2026", "autumn"],
        "relevant_source_titles": ["Application Deadlines"],
    },
    {
        "question": "Is there an online version of the MSADS program?",
        "expected_keywords": ["online", "working professionals"],
        "relevant_source_titles": ["Online Program"],
    },
    {
        "question": "Who are the faculty and instructors in the MSADS program?",
        "expected_keywords": ["instructor", "professor", "faculty"],
        "relevant_source_titles": ["Faculty and Admissions Contacts"],
    },
    {
        "question": "What is the difference between the 12-course and 18-course track?",
        "expected_keywords": ["12", "18", "thesis", "track"],
        "relevant_source_titles": ["Program Tracks and Structure"],
    },
    {
        "question": "What are some examples of past capstone projects?",
        "expected_keywords": ["capstone", "project", "industry"],
        "relevant_source_titles": ["Capstone Project"],
    },
    {
        "question": "How do I apply to the MSADS program?",
        "expected_keywords": ["application", "apply", "statement", "recommendation"],
        "relevant_source_titles": ["Admissions Requirements"],
    },
    {
        "question": "What programming experience do I need for MSADS?",
        "expected_keywords": ["python", "programming", "statistics", "mathematics"],
        "relevant_source_titles": ["Admissions Requirements", "Foundational Courses"],
    },
    {
        "question": "What is the online MSADS program designed for?",
        "expected_keywords": ["online", "working professionals", "experience"],
        "relevant_source_titles": ["Online Program"],
    },
    {
        "question": "Are there scholarships available for MSADS students?",
        "expected_keywords": ["scholarship", "merit", "financial"],
        "relevant_source_titles": ["Tuition and Financial Aid"],
    },
    {
        "question": "What student services are available in the MSADS program?",
        "expected_keywords": ["career", "student", "services"],
        "relevant_source_titles": ["In-Person Program Format", "Career Seminar"],
    },
    {
        "question": "Where are MSADS in-person classes held?",
        "expected_keywords": ["nbc tower", "chicago", "downtown", "gleacher"],
        "relevant_source_titles": ["In-Person Program Format"],
    },
    {
        "question": "What is the MBA/MS joint degree option?",
        "expected_keywords": ["mba", "booth", "joint", "degree"],
        "relevant_source_titles": ["Program Tracks and Structure", "Program Overview"],
    },
]
# ─────────────────────────────────────────────────────────────────────────────


def retrieval_precision_at_k(
    passages: List[Dict],
    relevant_titles: List[str],
    k: int = 4,
) -> float:
    """Fraction of top-K results matching expected source titles."""
    hits = sum(
        any(rt.lower() in p["title"].lower() for rt in relevant_titles)
        for p in passages[:k]
    )
    return hits / min(k, len(passages))


def mean_reciprocal_rank(
    passages: List[Dict],
    relevant_titles: List[str],
) -> float:
    """Reciprocal rank of the first relevant passage."""
    for rank, p in enumerate(passages, 1):
        if any(rt.lower() in p["title"].lower() for rt in relevant_titles):
            return 1.0 / rank
    return 0.0


def keyword_coverage(answer: str, keywords: List[str]) -> float:
    """Fraction of expected keywords present in the answer (case-insensitive)."""
    answer_lower = answer.lower()
    hits = sum(kw.lower() in answer_lower for kw in keywords)
    return hits / len(keywords)


def sentence_faithfulness(answer: str, context: str) -> float:
    """
    Lightweight Faithfulness (Method A — no external API needed).

    Algorithm:
    1. Split the answer into sentences.
    2. For each sentence, extract all meaningful phrases (4+ chars, skip
       stopwords and very common words).
    3. A sentence is considered 'grounded' if at least one of its key phrases
       appears in the retrieved context.
    4. Faithfulness = grounded sentences / total sentences.

    This is a conservative but transparent metric that requires no LLM calls.
    """
    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "this", "that", "these",
        "those", "i", "you", "he", "she", "it", "we", "they", "what", "which",
        "who", "whom", "when", "where", "why", "how", "and", "but", "or",
        "nor", "for", "so", "yet", "both", "either", "neither", "not", "only",
        "own", "same", "than", "too", "very", "just", "also", "about", "above",
        "after", "before", "between", "during", "from", "in", "into", "of",
        "on", "to", "with", "at", "by", "as", "if", "then", "than", "because",
        "while", "although", "however", "therefore", "thus", "hence",
    }

    context_lower = context.lower()

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if not sentences:
        return 0.0

    grounded = 0
    for sentence in sentences:
        words = re.findall(r'\b[a-z0-9][a-z0-9\-]*\b', sentence.lower())
        key_phrases = [w for w in words if w not in STOPWORDS and len(w) >= 4]

        # Check if any key phrase appears in the context
        if any(phrase in context_lower for phrase in key_phrases):
            grounded += 1

    return grounded / len(sentences)


def evaluate(store: MSADSVectorStore, verbose: bool = True) -> Dict:
    """
    Run all test questions through the retrieval pipeline and compute metrics.
    Does NOT call the LLM — faithfulness is computed against retrieved context only.
    """
    precision_scores = []
    mrr_scores = []
    faithfulness_scores = []
    keyword_scores = []

    for item in TEST_SET:
        passages = store.retrieve(item["question"], top_k=4)

        # Retrieval metrics
        precision_scores.append(
            retrieval_precision_at_k(passages, item["relevant_source_titles"])
        )
        mrr_scores.append(
            mean_reciprocal_rank(passages, item["relevant_source_titles"])
        )

        # Build context string from retrieved passages
        context = " ".join(p["text"] for p in passages)

        # Faithfulness: check how grounded the question itself is in context
        # (proxy metric — in production, use the actual LLM answer)
        faith_score = sentence_faithfulness(item["question"], context)
        faithfulness_scores.append(faith_score)

        # Keyword coverage (checks if expected keywords appear in context)
        kw_score = keyword_coverage(context, item["expected_keywords"])
        keyword_scores.append(kw_score)

    results = {
        "num_test_cases":       len(TEST_SET),
        "precision_at_4":       round(sum(precision_scores) / len(precision_scores), 4),
        "mean_reciprocal_rank": round(sum(mrr_scores) / len(mrr_scores), 4),
        "faithfulness":         round(sum(faithfulness_scores) / len(faithfulness_scores), 4),
        "keyword_coverage":     round(sum(keyword_scores) / len(keyword_scores), 4),
    }

    if verbose:
        print("\n" + "="*50)
        print("  MSADS RAG Evaluation Results")
        print("="*50)
        for k, v in results.items():
            label = k.replace("_", " ").title()
            print(f"  {label:<30} {v}")
        print("="*50)

    return results


if __name__ == "__main__":
    store = MSADSVectorStore()
    evaluate(store)
