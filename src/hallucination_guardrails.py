"""Retrieval-quality checks that prevent unsupported answer generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from src.rag_pipeline import RetrievedSource


SAFE_REFUSAL = "I don't know based on the indexed documents. I could not find enough relevant evidence to answer safely."


@dataclass(frozen=True)
class RetrievalDecision:
    """Whether retrieved evidence is strong enough to support generation."""

    allowed: bool
    qualifying_sources: int
    reason: str


def evaluate_retrieval(
    sources: Sequence[RetrievedSource],
    minimum_score: float = 0.75,
    minimum_chunks: int = 1,
) -> RetrievalDecision:
    """Refuse when too few retrieved chunks meet the relevance threshold."""
    if not 0 <= minimum_score <= 1:
        raise ValueError("minimum_score must be between 0 and 1")
    if minimum_chunks <= 0:
        raise ValueError("minimum_chunks must be positive")

    qualifying_sources = sum(source.score >= minimum_score for source in sources)
    if qualifying_sources < minimum_chunks:
        if not sources:
            reason = "retrieval returned no sources"
        else:
            reason = (
                f"only {qualifying_sources} source(s) reached the "
                f"{minimum_score:.2f} relevance threshold"
            )
        return RetrievalDecision(False, qualifying_sources, reason)

    return RetrievalDecision(True, qualifying_sources, "enough relevant evidence")
