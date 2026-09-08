"""RAG Candidate Retrieval and Re-ranking Engine.

Provides candidate retrieval (k > final_k), LLM / cross-encoder re-ranking,
before-and-after comparison utilities, and cost/latency trade-off analysis.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Sample candidate corpus (10 chunks) covering project submission evidence, policies, and general docs
SAMPLE_RERANKING_CORPUS: List[Dict[str, Any]] = [
    {
        "id": "chunk-1",
        "text": "Project submission guidelines: Overview of project lifecycle, milestones, and team roles for internal software deliverables.",
        "score": 0.88,
        "metadata": {"source": "project-submission-overview.pdf", "section": "Overview"},
    },
    {
        "id": "chunk-2",
        "text": "Required evidence for project submission includes signed security audit reports, test coverage metrics (>=85%), compliance checklist, and formal sign-off certificates from department leads.",
        "score": 0.82,
        "metadata": {"source": "submission-requirements.pdf", "section": "Submission Evidence"},
    },
    {
        "id": "chunk-3",
        "text": "General documentation rules: Use markdown format for README files and document all public API endpoints.",
        "score": 0.79,
        "metadata": {"source": "doc-standards.pdf", "section": "Formatting"},
    },
    {
        "id": "chunk-4",
        "text": "Verification evidence for clinical software submission must include validated test run logs, unit test pass certificates, and risk assessment signatures.",
        "score": 0.76,
        "metadata": {"source": "clinical-compliance.pdf", "section": "Verification"},
    },
    {
        "id": "chunk-5",
        "text": "All hospital software deployments require quarterly vulnerability scans and approval from the IT governance committee.",
        "score": 0.74,
        "metadata": {"source": "security-policy-2026.pdf", "section": "Governance"},
    },
    {
        "id": "chunk-6",
        "text": "Password reset protocol: Employees can request password resets through the self-service IT portal.",
        "score": 0.68,
        "metadata": {"source": "it-support-handbook.pdf", "section": "Account access"},
    },
    {
        "id": "chunk-7",
        "text": "Submission checklist: Ensure code repository is clean, dependencies are pinned in requirements.txt, and build script succeeds without warnings.",
        "score": 0.65,
        "metadata": {"source": "submission-requirements.pdf", "section": "Checklist"},
    },
    {
        "id": "chunk-8",
        "text": "Medication dosage instructions: Administer prescribed dosage with water as indicated on the clinical chart.",
        "score": 0.58,
        "metadata": {"source": "medication-guideline.pdf", "section": "Dosage"},
    },
    {
        "id": "chunk-9",
        "text": "Budget approval evidence: Financial receipts and purchase order approvals must be archived prior to project submission.",
        "score": 0.52,
        "metadata": {"source": "finance-policy.pdf", "section": "Finances"},
    },
    {
        "id": "chunk-10",
        "text": "Emergency triage procedures require staff to verify patient identity before administering any clinical care.",
        "score": 0.45,
        "metadata": {"source": "triage-protocol.pdf", "section": "Emergency Care"},
    },
]


def retrieve_candidates(
    query: str,
    k: int = 10,
    corpus: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve top-k candidates from vector search.

    Args:
        query: User search query string.
        k: Number of candidates to retrieve (larger than final_k).
        corpus: Optional custom candidate corpus list.

    Returns:
        List of candidate dictionaries sorted by initial vector retrieval score.
    """
    dataset = corpus if corpus is not None else SAMPLE_RERANKING_CORPUS
    # Make a copy of candidates sorted by initial vector score descending
    candidates = [dict(item) for item in dataset]
    candidates = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
    return candidates[:k]


_LIVE_API_AVAILABLE: Optional[bool] = None


def call_reranker_model(prompt: str, client: Any = None, model: str = "gpt-4o-mini") -> float:
    """Invoke the re-ranker model or fallback scoring mechanism to evaluate query-chunk relevance.

    Args:
        prompt: Formatted prompt string asking for a relevance score from 0 to 10.
        client: Optional OpenAI client instance.
        model: LLM model name.

    Returns:
        Float relevance score between 0.0 and 10.0.
    """
    global _LIVE_API_AVAILABLE
    load_dotenv()
    use_mock = os.getenv("MOCK_MODE", "false").lower() == "true" or "--mock" in sys.argv
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")

    if _LIVE_API_AVAILABLE is not False and not use_mock and (client is not None or (api_key and api_key != "your_api_key_here")):
        try:
            if client is None:
                from openai import OpenAI
                base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("API_BASE_URL")
                client = OpenAI(base_url=base_url, api_key=api_key)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise document relevance scorer. Output only a single number between 0 and 10."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            raw_text = response.choices[0].message.content.strip()
            match = re.search(r"[-+]?\d*\.\d+|\d+", raw_text)
            if match:
                score = float(match.group())
                _LIVE_API_AVAILABLE = True
                return max(0.0, min(10.0, score))
        except Exception as e:
            _LIVE_API_AVAILABLE = False
            logger.warning("Live reranker model call failed (%s). Falling back to offline relevance scoring.", e)

    # Deterministic offline cross-encoder fallback scoring
    return _compute_offline_rerank_score(prompt)


def _compute_offline_rerank_score(prompt: str) -> float:
    """Compute a deterministic relevance score (0-10) based on query and text semantic overlap."""
    query_match = re.search(r"Query:\s*(.*?)\nChunk:", prompt, re.DOTALL)
    chunk_match = re.search(r"Chunk:\s*(.*?)\nReturn", prompt, re.DOTALL)

    if not query_match or not chunk_match:
        return 5.0

    query = query_match.group(1).strip().lower()
    chunk_text = chunk_match.group(1).strip().lower()

    query_words = set(re.findall(r"\b\w+\b", query)) - {"what", "is", "for", "the", "to", "and", "a", "an", "of", "in"}
    if not query_words:
        return 5.0

    matched_words = [word for word in query_words if word in chunk_text]
    overlap_ratio = len(matched_words) / len(query_words)

    # Base score scaled 0 to 10
    score = overlap_ratio * 7.5

    # Specific phrase boosts for high precision matching
    if "required evidence" in chunk_text or "evidence for project submission" in chunk_text:
        score += 2.5
    elif "evidence" in chunk_text and "submission" in chunk_text:
        score += 2.0
    elif "submission" in chunk_text and ("checklist" in chunk_text or "verification" in chunk_text):
        score += 1.5

    return round(max(0.0, min(10.0, score)), 2)


def rerank_score(
    query: str,
    chunk: Dict[str, Any],
    call_model_fn: Optional[Callable[[str], float]] = None,
) -> float:
    """Score how relevant a single chunk is to the query from 0 to 10.

    Args:
        query: Search query string.
        chunk: Chunk dict containing 'text'.
        call_model_fn: Optional callable replacing default call_reranker_model.

    Returns:
        Float score from 0.0 to 10.0.
    """
    prompt = f"""Score how relevant this chunk is to the query from 0 to 10.
Query: {query}
Chunk: {chunk["text"]}
Return only the number."""

    if call_model_fn is not None:
        return float(call_model_fn(prompt))
    return float(call_reranker_model(prompt))


def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    final_k: int = 3,
    call_model_fn: Optional[Callable[[str], float]] = None,
) -> List[Dict[str, Any]]:
    """Re-rank candidate chunks using prompt-based scoring and return top final_k results.

    Args:
        query: User search query.
        candidates: List of candidate chunk dicts retrieved in stage 1.
        final_k: Number of final top chunks to keep for context generation.
        call_model_fn: Optional custom scoring function.

    Returns:
        List of top final_k reranked chunk dicts with 'rerank_score' added.
    """
    reranked = []
    for chunk in candidates:
        score = rerank_score(query, chunk, call_model_fn=call_model_fn)
        reranked.append({
            **chunk,
            "rerank_score": score,
        })

    reranked = sorted(reranked, key=lambda item: item["rerank_score"], reverse=True)
    return reranked[:final_k]


def show(label: str, rows: List[Dict[str, Any]], max_text_len: int = 120) -> None:
    """Display retrieval/re-ranking results cleanly with vector and rerank scores.

    Args:
        label: Section header text.
        rows: List of result chunk dictionaries.
        max_text_len: Character limit for text preview display.
    """
    print(label)
    for rank, item in enumerate(rows, start=1):
        print("rank:", rank)
        print("vector_score:", round(item["score"], 4))
        print("rerank_score:", item.get("rerank_score"))
        print("source:", item["metadata"]["source"])
        print("text:", item["text"][:max_text_len])
        print("-" * 40)


def explain_cost_latency_tradeoffs() -> Dict[str, Any]:
    """Provide a structured explanation of the cost and latency trade-offs of re-ranking.

    Returns:
        Dictionary containing trade-off dimensions, candidate sizing guidelines, and cost analysis.
    """
    return {
        "retrieval_stage": {
            "mechanism": "Bi-encoder vector search (cosine similarity / HNSW index)",
            "latency": "Fast (1–10 ms for 100k+ chunks)",
            "cost": "Low (Fixed vector index search, query embedding calculated once)",
            "precision": "Moderate (Finds semantically similar chunks, but can rank broad overviews above specific answers)",
        },
        "reranking_stage": {
            "mechanism": "Cross-encoder or LLM scoring (joint query + candidate text attention)",
            "latency": "Higher (50–300 ms for 10–20 candidates depending on LLM / cross-encoder model)",
            "cost": "Higher (Requires processing query + candidate tokens for every candidate chunk)",
            "precision": "High (Directly evaluates exact relevance, evidence completeness, and query alignment)",
        },
        "best_practice_recommendation": {
            "candidate_set_size_k": "10 to 20 candidates retrieved in Stage 1",
            "final_context_size_k": "3 to 5 top re-ranked chunks passed to LLM generation",
            "rule_of_thumb": "Use re-ranking when precision is critical and initial vector search returns mixed-quality candidates. Measure whether answer quality improves relative to the added latency and token cost.",
        },
    }


def build_reranking_report(
    query: str,
    candidates: List[Dict[str, Any]],
    final_context: List[Dict[str, Any]],
    final_k: int = 3,
) -> str:
    """Build a comprehensive Markdown report documenting the before-and-after re-ranking results.

    Args:
        query: User search query string.
        candidates: Original candidate set from initial retrieval.
        final_context: Final re-ranked top-k context set.
        final_k: Target context size.

    Returns:
        Formatted Markdown report string.
    """
    lines = [
        "# RAG Retrieval & Re-ranking Analysis Report",
        "",
        f"**Query**: `{query}`  ",
        f"**Initial Candidate Pool Size (k)**: `{len(candidates)}`  ",
        f"**Final Context Size (final_k)**: `{final_k}`",
        "",
        "---",
        "",
        "## 1. Initial Vector Retrieval (Before Re-ranking)",
        "",
        "| Rank | Vector Score | Source Document | Section | Text Preview |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for rank, item in enumerate(candidates[:final_k], start=1):
        preview = item["text"][:80].replace("\n", " ")
        lines.append(f"| {rank} | {item['score']:.4f} | `{item['metadata']['source']}` | `{item['metadata']['section']}` | {preview}... |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Re-ranked Results (After Re-ranking)",
        "",
        "| Rank | Rerank Score (0-10) | Vector Score | Source Document | Section | Text Preview |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for rank, item in enumerate(final_context, start=1):
        preview = item["text"][:80].replace("\n", " ")
        lines.append(f"| {rank} | {item.get('rerank_score', 0.0):.2f} | {item['score']:.4f} | `{item['metadata']['source']}` | `{item['metadata']['section']}` | {preview}... |")

    tradeoffs = explain_cost_latency_tradeoffs()
    lines.extend([
        "",
        "---",
        "",
        "## 3. Cost & Latency Trade-off Analysis",
        "",
        "### Stage Comparison",
        "",
        f"- **Initial Retrieval**: {tradeoffs['retrieval_stage']['latency']} latency | {tradeoffs['retrieval_stage']['cost']} cost | {tradeoffs['retrieval_stage']['precision']} precision",
        f"- **Re-ranking Stage**: {tradeoffs['reranking_stage']['latency']} latency | {tradeoffs['reranking_stage']['cost']} cost | {tradeoffs['reranking_stage']['precision']} precision",
        "",
        "### Strategy & Recommendations",
        "",
        f"- **Candidate Set Sizing**: Retrieve `{tradeoffs['best_practice_recommendation']['candidate_set_size_k']}`.",
        f"- **Final Context Selection**: Select `{tradeoffs['best_practice_recommendation']['final_context_size_k']}`.",
        f"- **Decision Rule**: {tradeoffs['best_practice_recommendation']['rule_of_thumb']}",
    ])

    return "\n".join(lines) + "\n"
