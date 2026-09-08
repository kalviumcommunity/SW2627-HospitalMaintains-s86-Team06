"""Metadata Filtering and Hybrid Search Module.

Provides functions for restricting vector retrieval using metadata filters,
comparing filtered and unfiltered search results, keyword scoring,
and hybrid ranking (combining semantic vector similarity with lexical keyword matching).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

from src.embedding_demo import (
    OFFLINE_QUERY_VECTOR,
    OFFLINE_VECTORS,
    TextChunk,
    cosine_similarity,
)
from src.vector_db import VectorCollection

logger = logging.getLogger(__name__)

# Sample corpus designed to demonstrate metadata filtering and hybrid search
DEFAULT_METADATA_CORPUS: List[TextChunk] = [
    TextChunk(
        "To reset your employee password, navigate to the self-service IT help desk portal and follow the password reset prompt.",
        {
            "source": "it-support-handbook.pdf",
            "section": "Account access",
            "department": "IT Support",
            "access_level": "all_staff",
        },
    ),
    TextChunk(
        "All hospital network accounts require a password reset every 90 days according to the corporate security policy.",
        {
            "source": "security-policy-2026.pdf",
            "section": "Security policy",
            "department": "Cybersecurity",
            "access_level": "admin",
        },
    ),
    TextChunk(
        "Campus IT infrastructure maintains wireless routers and account provisioning for campus facilities.",
        {
            "source": "campus-it-guide.pdf",
            "section": "Campus IT",
            "department": "IT Operations",
            "access_level": "all_staff",
        },
    ),
    TextChunk(
        "The patient should take the prescribed medication with water following post-surgery clinical guidelines.",
        {
            "source": "medication-guideline.pdf",
            "section": "Dosage",
            "department": "Pharmacy",
            "access_level": "medical_staff",
        },
    ),
    TextChunk(
        "Emergency triage procedures require staff to verify patient identity before administering any clinical care.",
        {
            "source": "triage-protocol.pdf",
            "section": "Emergency Care",
            "department": "Emergency",
            "access_level": "medical_staff",
        },
    ),
]

# Deterministic vectors for the sample metadata corpus
DEFAULT_METADATA_VECTORS: List[List[float]] = [
    [0.05, 0.02, 0.92, 0.04, 0.08, 0.02, 0.01, 0.05],  # Account access (IT password reset steps)
    [0.06, 0.03, 0.85, 0.05, 0.09, 0.03, 0.01, 0.04],  # Security policy (IT password policy 90 days)
    [0.08, 0.04, 0.78, 0.06, 0.10, 0.04, 0.02, 0.05],  # Campus IT (wireless/account)
    [0.91, 0.08, 0.02, 0.12, 0.04, 0.03, 0.01, 0.02],  # Dosage (medication)
    [0.85, 0.12, 0.04, 0.15, 0.05, 0.02, 0.01, 0.03],  # Emergency Care
]


def retrieve(
    query: str,
    k: int = 3,
    metadata_filter: Optional[Dict[str, Any]] = None,
    records: Optional[Sequence[Dict[str, Any]]] = None,
    query_vector: Optional[Sequence[float]] = None,
    collection: Optional[VectorCollection] = None,
) -> List[Dict[str, Any]]:
    """Retrieve top-k documents using vector search with optional metadata filtering.

    Supports querying both an in-memory record list and a ChromaDB VectorCollection wrapper.

    Args:
        query: Query string text.
        k: Number of top results to return.
        metadata_filter: Optional metadata key-value filter dictionary (e.g. {"section": "Account access"}).
        records: Optional sequence of stored record dicts with 'text', 'metadata', and 'embedding'.
        query_vector: Optional pre-computed query embedding vector.
        collection: Optional VectorCollection instance (ChromaDB collection wrapper).

    Returns:
        List of dicts containing score, text, metadata, and id.
    """
    if collection is not None:
        if query_vector is None:
            query_vector = OFFLINE_QUERY_VECTOR
        results = collection.query(
            query_vector=query_vector,
            n_results=k,
            where=metadata_filter,
        )
        # Standardize result keys to match search result format: score = 1.0 - distance (or similarity)
        formatted: List[Dict[str, Any]] = []
        for item in results:
            distance = item.get("distance")
            score = 1.0 - float(distance) if distance is not None else 1.0
            formatted.append(
                {
                    "id": item.get("id", ""),
                    "score": score,
                    "text": item.get("text", ""),
                    "metadata": item.get("metadata", {}),
                }
            )
        return formatted

    if records is None:
        records = [
            {
                "id": f"chunk-{i}",
                "text": chunk.text,
                "metadata": chunk.metadata,
                "embedding": DEFAULT_METADATA_VECTORS[i],
            }
            for i, chunk in enumerate(DEFAULT_METADATA_CORPUS)
        ]

    if query_vector is None:
        query_vector = OFFLINE_QUERY_VECTOR

    # Filter records based on metadata_filter before or during vector ranking
    filtered_records: List[Dict[str, Any]] = []
    for rec in records:
        if metadata_filter:
            matches = True
            rec_meta = rec.get("metadata", {})
            for key, expected_val in metadata_filter.items():
                if rec_meta.get(key) != expected_val:
                    matches = False
                    break
            if not matches:
                continue
        filtered_records.append(rec)

    # Calculate similarity score for filtered records
    scored_results: List[Dict[str, Any]] = []
    for rec in filtered_records:
        sim_score = cosine_similarity(query_vector, rec["embedding"])
        scored_results.append(
            {
                "id": rec.get("id", ""),
                "score": float(sim_score),
                "text": rec.get("text", ""),
                "metadata": rec.get("metadata", {}),
            }
        )

    scored_results.sort(key=lambda item: item["score"], reverse=True)
    return scored_results[:k]


def show_results(label: str, results: List[Dict[str, Any]]) -> None:
    """Print result sets to compare sources, scores, sections, and text snippets."""
    print(label)
    for item in results:
        print("score:", round(item["score"], 4))
        print("source:", item["metadata"].get("source", "N/A"))
        print("section:", item["metadata"].get("section", "N/A"))
        print("text:", item["text"][:120])
        print("-" * 40)


def keyword_score(text: str, keywords: Sequence[str]) -> int:
    """Compute lexical keyword score based on exact term matches in text.

    Args:
        text: Target passage text.
        keywords: List of keyword strings to search for.

    Returns:
        Integer count of keywords found in text (case-insensitive).
    """
    lowered = text.lower()
    return sum(1 for word in keywords if word.lower() in lowered)


def hybrid_rank(
    vector_results: List[Dict[str, Any]],
    keywords: Sequence[str],
    vector_weight: float = 0.8,
    keyword_weight: float = 0.2,
) -> List[Dict[str, Any]]:
    """Combine vector similarity scores with lexical keyword scores to re-rank results.

    Args:
        vector_results: List of result dicts from vector search (containing 'score', 'text', 'metadata').
        keywords: Sequence of keyword terms to match lexically.
        vector_weight: Weight multiplier for vector similarity score (default 0.8).
        keyword_weight: Weight multiplier for lexical keyword score (default 0.2).

    Returns:
        List of result dicts sorted by combined 'hybrid_score' in descending order.
    """
    ranked: List[Dict[str, Any]] = []
    for item in vector_results:
        lexical = keyword_score(item["text"], keywords)
        combined = (vector_weight * item["score"]) + (keyword_weight * lexical)
        ranked.append(
            {
                **item,
                "keyword_score": lexical,
                "hybrid_score": combined,
            }
        )
    return sorted(ranked, key=lambda item: item["hybrid_score"], reverse=True)


def build_metadata_hybrid_report(
    query: str,
    unfiltered_results: List[Dict[str, Any]],
    filtered_results: List[Dict[str, Any]],
    hybrid_results: List[Dict[str, Any]],
    metadata_filter: Dict[str, Any],
    keywords: Sequence[str],
) -> str:
    """Build a comprehensive Markdown report documenting metadata filtering and hybrid search."""
    lines = [
        "# Metadata Filtering & Hybrid Search Report",
        "",
        f"Query: **\"{query}\"**",
        f"Metadata Filter Applied: `{metadata_filter}`",
        f"Hybrid Keywords: `{list(keywords)}`",
        "",
        "## Executive Summary",
        "",
        "- **Metadata Filtering**: Restricts retrieval scope before/during vector search to records matching specific metadata criteria (e.g. section, source, department, date, access level).",
        "- **Semantic Vector Search vs Lexical Keyword Search**:",
        "  - *Vector Search (Semantic)*: Matches meaning even when exact words differ using high-dimensional embeddings.",
        "  - *Keyword Search (Lexical)*: Rewards exact word, name, product ID, or code occurrences.",
        "  - *Hybrid Search*: Combines semantic similarity scores with lexical match weights to optimize overall precision.",
        "",
        "## Unfiltered Vector Retrieval",
        "",
        "Unfiltered queries search the entire corpus. While semantic vector search surfaces high-similarity chunks, plausible-sounding chunks from unrelated sections (e.g. general security policy or campus IT) may dilute precision.",
        "",
    ]

    for idx, item in enumerate(unfiltered_results, 1):
        lines.extend(
            [
                f"{idx}. **Score:** `{item['score']:.4f}` | **Source:** `{item['metadata'].get('source', 'N/A')}` | **Section:** `{item['metadata'].get('section', 'N/A')}`",
                f"   **Text:** {item['text']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Metadata-Filtered Retrieval",
            "",
            f"Filtering by `{metadata_filter}` ensures that retrieval targets only chunks explicitly belonging to the intended scope. Unrelated sections are pruned out, dramatically raising precision.",
            "",
        ]
    )

    for idx, item in enumerate(filtered_results, 1):
        lines.extend(
            [
                f"{idx}. **Score:** `{item['score']:.4f}` | **Source:** `{item['metadata'].get('source', 'N/A')}` | **Section:** `{item['metadata'].get('section', 'N/A')}`",
                f"   **Text:** {item['text']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Hybrid Lexical-Semantic Search",
            "",
            "Hybrid search combines vector scores (`vector_weight=0.8`) with keyword match counts (`keyword_weight=0.2`). This preserves semantic ranking while guaranteeing top priority for exact term matches.",
            "",
        ]
    )

    for idx, item in enumerate(hybrid_results, 1):
        lines.extend(
            [
                f"{idx}. **Hybrid Score:** `{item['hybrid_score']:.4f}` (Vector: `{item['score']:.4f}`, Keyword Matches: `{item['keyword_score']}`)",
                f"   **Source:** `{item['metadata'].get('source', 'N/A')}` | **Section:** `{item['metadata'].get('section', 'N/A')}`",
                f"   **Text:** {item['text']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Trade-offs & Strategic Guidance",
            "",
            "1. **Precision vs Recall**:",
            "   - Metadata filtering **improves precision** by eliminating plausible but wrong areas of the corpus.",
            "   - Overly strict filters or missing metadata tags can **hurt recall** by excluding relevant passages.",
            "2. **When to Use Hybrid Search**:",
            "   - Highly recommended when users query exact product names, policy IDs, error codes, course codes, or names where exact-match precision is essential.",
            "",
        ]
    )

    return "\n".join(lines)
