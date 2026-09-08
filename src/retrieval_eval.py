"""Quantitative Retrieval Quality Evaluation Engine.

Provides labelled query dataset evaluation, recall@k and precision@k computation,
failure inspection diagnostics, and quality report generation.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Sample labelled corpus mapping chunk IDs to content
SAMPLE_LABELLED_CORPUS: List[Dict[str, Any]] = [
    {
        "id": "account-guide.md:0",
        "text": "How to reset your employee password: navigate to the self-service IT help desk portal and follow the password reset prompt.",
        "metadata": {"source": "account-guide.md", "section": "Account access"},
    },
    {
        "id": "account-guide.md:1",
        "text": "Password reset steps for learners: verify your employee ID and submit security challenge questions.",
        "metadata": {"source": "account-guide.md", "section": "Password Recovery"},
    },
    {
        "id": "submission-rubric.md:2",
        "text": "Required evidence for project submission includes signed security audit reports, test coverage metrics (>=85%), compliance checklist, and formal sign-off certificates.",
        "metadata": {"source": "submission-rubric.md", "section": "Submission Evidence"},
    },
    {
        "id": "medication-guideline.pdf:0",
        "text": "The patient should take the prescribed medication with water following post-surgery clinical guidelines.",
        "metadata": {"source": "medication-guideline.pdf", "section": "Dosage"},
    },
    {
        "id": "triage-protocol.pdf:0",
        "text": "Emergency triage procedures require staff to verify patient identity before administering any clinical care.",
        "metadata": {"source": "triage-protocol.pdf", "section": "Emergency Care"},
    },
    {
        "id": "campus-it-guide.pdf:0",
        "text": "Campus IT infrastructure maintains wireless routers and account provisioning for campus facilities.",
        "metadata": {"source": "campus-it-guide.pdf", "section": "Campus IT"},
    },
    {
        "id": "security-policy.pdf:5",
        "text": "Hospital corporate security policy dictates that all network passcodes expire every 90 days.",
        "metadata": {"source": "security-policy.pdf", "section": "Security Policy"},
    },
]

# Standard labelled queries mapping queries to expected relevant chunk IDs
LABELLED_QUERIES: List[Dict[str, Any]] = [
    {
        "query": "How can a learner reset their password?",
        "relevant_chunk_ids": {"account-guide.md:0", "account-guide.md:1"},
    },
    {
        "query": "What evidence is required for project submission?",
        "relevant_chunk_ids": {"submission-rubric.md:2"},
    },
    {
        "query": "What medication instructions should the patient follow?",
        "relevant_chunk_ids": {"medication-guideline.pdf:0"},
    },
    {
        "query": "What emergency triage verification procedures exist?",
        "relevant_chunk_ids": {"triage-protocol.pdf:0"},
    },
    {
        "query": "What are the rules for campus wireless network account provisioning?",
        "relevant_chunk_ids": {"campus-it-guide.pdf:0"},
    },
]


def default_retrieve(query: str, k: int = 5, corpus: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Simple deterministic term-overlap & semantic vector retrieval for demonstration.

    Args:
        query: User search query.
        k: Top-k count.
        corpus: Optional custom corpus list.

    Returns:
        List of retrieved chunk dictionaries sorted by relevance score.
    """
    dataset = corpus if corpus is not None else SAMPLE_LABELLED_CORPUS
    query_words = set(re.findall(r"\b\w+\b", query.lower())) - {"what", "how", "is", "are", "the", "for", "a", "an", "to", "their"}

    scored_chunks = []
    for chunk in dataset:
        text = chunk["text"].lower()
        chunk_words = set(re.findall(r"\b\w+\b", text))
        matched = query_words.intersection(chunk_words)
        score = len(matched) / (len(query_words) + 0.1)

        # Bonus keyword phrase matches
        if "reset" in query.lower() and "reset" in text:
            score += 0.5
        if "evidence" in query.lower() and "evidence" in text:
            score += 0.5

        scored_chunks.append({**chunk, "score": round(score, 4)})

    scored_chunks = sorted(scored_chunks, key=lambda x: x["score"], reverse=True)
    return scored_chunks[:k]


def evaluate_query(
    item: Dict[str, Any],
    k: int = 5,
    retrieve_fn: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Evaluate retrieval performance for a single labelled query item.

    Args:
        item: Labelled query item dictionary containing 'query' and 'relevant_chunk_ids'.
        k: Top-k retrieval cut-off.
        retrieve_fn: Custom retrieval function (query, k) -> List[Dict].

    Returns:
        Dictionary containing retrieved_ids, hits, recall, and precision.
    """
    fn = retrieve_fn if retrieve_fn is not None else default_retrieve
    results = fn(item["query"], k=k)
    retrieved_ids = [result["id"] for result in results]
    relevant = set(item["relevant_chunk_ids"])

    hits = [chunk_id for chunk_id in retrieved_ids if chunk_id in relevant]
    recall = len(hits) / len(relevant) if relevant else 0.0
    precision = len(hits) / len(retrieved_ids) if retrieved_ids else 0.0

    return {
        "query": item["query"],
        "retrieved_ids": retrieved_ids,
        "relevant_chunk_ids": sorted(list(relevant)),
        "hits": hits,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
    }


def evaluate_dataset(
    labelled_queries: List[Dict[str, Any]],
    k: int = 5,
    retrieve_fn: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Evaluate retrieval performance across a full dataset of labelled queries.

    Args:
        labelled_queries: List of labelled query dicts.
        k: Top-k retrieval cut-off.
        retrieve_fn: Custom retrieval function.

    Returns:
        Structured dictionary with individual query rows, avg_recall, avg_precision, and failures.
    """
    rows = [evaluate_query(item, k=k, retrieve_fn=retrieve_fn) for item in labelled_queries]

    avg_recall = sum(row["recall"] for row in rows) / len(rows) if rows else 0.0
    avg_precision = sum(row["precision"] for row in rows) / len(rows) if rows else 0.0
    failures = [row for row in rows if row["recall"] < 1.0]

    return {
        "k": k,
        "total_queries": len(rows),
        "avg_recall": round(avg_recall, 4),
        "avg_precision": round(avg_precision, 4),
        "rows": rows,
        "failures": failures,
    }


def inspect_failures(failures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analyze failed retrieval queries (recall < 1.0) and identify likely root causes.

    Args:
        failures: List of failure evaluation rows.

    Returns:
        List of structured failure analysis dicts with query, missing chunk IDs, and diagnostic reason.
    """
    diagnostics = []
    for failure in failures:
        missing_ids = set(failure["relevant_chunk_ids"]) - set(failure["retrieved_ids"])
        
        if not failure["retrieved_ids"]:
            reason = "No candidates retrieved from vector index (empty result set)."
        elif len(missing_ids) == len(failure["relevant_chunk_ids"]):
            reason = "Complete retrieval miss: embedding mismatch, vocabulary gap, or harsh metadata filter."
        else:
            reason = "Partial retrieval hit: k is too small to capture all relevant chunks, or chunking split relevant context."

        diagnostics.append({
            "query": failure["query"],
            "expected": failure["relevant_chunk_ids"],
            "retrieved": failure["retrieved_ids"],
            "missing_ids": sorted(list(missing_ids)),
            "recall": failure["recall"],
            "precision": failure["precision"],
            "root_cause_diagnosis": reason,
        })
    return diagnostics


def explain_improvement_strategies() -> Dict[str, str]:
    """Explain strategic options to improve recall and precision when retrieval failures occur.

    Returns:
        Dictionary mapping improvement technique to operational rationale.
    """
    return {
        "increase_k": "Increase top-k candidate parameter (e.g. from 5 to 10 or 20) so borderline relevant chunks enter the candidate set.",
        "hybrid_search": "Combine sparse BM25 keyword matching with dense vector search to bridge vocabulary gaps (e.g., specific ID or code queries).",
        "metadata_filtering": "Apply strict section or source metadata filters to eliminate out-of-domain noise from the vector search space.",
        "re_ranking": "Use a two-stage pipeline: retrieve a large candidate set (k=20) and apply a cross-encoder / LLM re-ranker to boost relevant chunks to top ranks.",
        "chunking_tuning": "Adjust chunk sizes and overlap to preserve complete context windows without fragmenting key facts across chunk boundaries.",
        "query_rewriting": "Use LLM query expansion / HyDE (Hypothetical Document Embeddings) to expand short or ambiguous user queries.",
    }


def build_evaluation_report(eval_summary: Dict[str, Any]) -> str:
    """Build a comprehensive Markdown report documenting retrieval evaluation metrics and failure inspection.

    Args:
        eval_summary: Output dictionary from evaluate_dataset.

    Returns:
        Formatted Markdown report string.
    """
    k = eval_summary["k"]
    rows = eval_summary["rows"]
    failures = eval_summary["failures"]
    avg_recall = eval_summary["avg_recall"]
    avg_precision = eval_summary["avg_precision"]

    lines = [
        "# Quantitative Retrieval Quality Evaluation Report",
        "",
        f"**Queries Evaluated**: `{eval_summary['total_queries']}`  ",
        f"**Top-k Cutoff**: `k = {k}`  ",
        f"**Recall@{k}**: `{avg_recall:.4f}` (`{avg_recall * 100:.1f}%`)  ",
        f"**Precision@{k}**: `{avg_precision:.4f}` (`{avg_precision * 100:.1f}%`)  ",
        f"**Failure Count (Recall < 1.0)**: `{len(failures)}`",
        "",
        "---",
        "",
        "## 1. Query-Level Evaluation Results",
        "",
        "| Query | Expected Chunk IDs | Retrieved Chunk IDs | Hits | Recall | Precision |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for row in rows:
        expected_str = "<br>".join(row["relevant_chunk_ids"])
        retrieved_str = "<br>".join(row["retrieved_ids"][:3]) + ("..." if len(row["retrieved_ids"]) > 3 else "")
        hits_str = ", ".join(row["hits"]) if row["hits"] else "None"
        lines.append(
            f"| `{row['query']}` | {expected_str} | {retrieved_str} | `{hits_str}` | {row['recall']:.2f} | {row['precision']:.2f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Failure Inspection & Root Cause Diagnostics",
        "",
    ])

    if failures:
        diagnostics = inspect_failures(failures)
        for idx, diag in enumerate(diagnostics, start=1):
            lines.extend([
                f"### Failure #{idx}: `{diag['query']}`",
                "",
                f"- **Expected Chunk IDs**: `{diag['expected']}`",
                f"- **Retrieved Chunk IDs**: `{diag['retrieved']}`",
                f"- **Missing Chunk IDs**: `{diag['missing_ids']}`",
                f"- **Metrics**: Recall = `{diag['recall']:.2f}`, Precision = `{diag['precision']:.2f}`",
                f"- **Root Cause Diagnosis**: {diag['root_cause_diagnosis']}",
                "",
            ])
    else:
        lines.append("✅ **No failures observed!** All labelled queries achieved 100% recall@k.")

    strategies = explain_improvement_strategies()
    lines.extend([
        "",
        "---",
        "",
        "## 3. Actionable Strategies to Improve Recall",
        "",
    ])

    for strategy, desc in strategies.items():
        lines.append(f"- **{strategy.replace('_', ' ').title()}**: {desc}")

    return "\n".join(lines) + "\n"
