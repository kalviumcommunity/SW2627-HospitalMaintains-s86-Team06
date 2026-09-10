"""Citation records that map generated answer markers to retrieved chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class SourceCitation:
    """A verifiable answer marker and the retrieved evidence behind it."""

    marker: str
    text: str
    metadata: dict[str, Any]
    score: float

    @property
    def source_document(self) -> str:
        return str(self.metadata.get("source_document", "unknown source"))


def build_citations(sources: Sequence[Any]) -> list[SourceCitation]:
    """Assign markers in answer order to the exact retrieved source chunks."""
    return [
        SourceCitation(
            marker=f"[{index}]",
            text=source.text,
            metadata=dict(source.metadata),
            score=float(source.score),
        )
        for index, source in enumerate(sources, start=1)
    ]


def verify_citation(citation: SourceCitation, retrieved_text: str) -> bool:
    """Verify that a citation still points to its original retrieved text."""
    return bool(retrieved_text) and citation.text == retrieved_text


def citation_mapping(citations: Sequence[SourceCitation]) -> list[dict[str, Any]]:
    """Return a serializable marker-to-source mapping for audit or display."""
    return [
        {
            "marker": citation.marker,
            "source_document": citation.source_document,
            "metadata": citation.metadata,
            "score": citation.score,
            "retrieved_text": citation.text,
        }
        for citation in citations
    ]
