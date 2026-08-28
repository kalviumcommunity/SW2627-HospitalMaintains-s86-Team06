"""Generate and compare a few text embeddings.

With an OpenAI client this calls the configured embeddings model. Without one,
the small deterministic semantic fixture keeps the educational demo runnable
offline; it is not a replacement for a production embedding model.
"""

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class TextChunk:
    text: str
    metadata: dict[str, Any]


SAMPLE_CORPUS = [
    TextChunk(
        "The patient should take the prescribed medication with water.",
        {"source_document": "medication-guideline.pdf", "chunk_index": 1, "section": "Dosage"},
    ),
    TextChunk(
        "Patients need to use their recommended medicine with water.",
        {"source_document": "medication-guideline.pdf", "chunk_index": 2, "section": "Administration"},
    ),
    TextChunk(
        "The help desk can reset an employee password.",
        {"source_document": "it-support-handbook.pdf", "chunk_index": 1, "section": "Account access"},
    ),
]

SAMPLE_TEXTS = [chunk.text for chunk in SAMPLE_CORPUS]
SAMPLE_QUERY = "What medication instructions should the patient follow?"

# Deterministic semantic fixture for environments without API credentials.
OFFLINE_VECTORS = [
    [0.91, 0.08, 0.02, 0.12, 0.04, 0.03, 0.01, 0.02],
    [0.89, 0.10, 0.03, 0.11, 0.05, 0.02, 0.01, 0.03],
    [0.04, 0.02, 0.91, 0.03, 0.08, 0.02, 0.01, 0.05],
]
OFFLINE_QUERY_VECTOR = [0.90, 0.09, 0.02, 0.11, 0.04, 0.03, 0.01, 0.02]


def cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    """Return cosine similarity for two equal-length vectors."""
    if len(first) != len(second):
        raise ValueError("Vectors must have the same length")
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector")
    return sum(a * b for a, b in zip(first, second)) / (first_norm * second_norm)


def generate_embeddings(
    texts: Sequence[str], client: Any | None = None, model: str | None = None
) -> tuple[str, list[list[float]]]:
    """Generate vectors through an embeddings API or the explicit offline fixture."""
    if client is not None:
        if not model:
            raise ValueError("An embedding model is required when using an embeddings API")
        response = client.embeddings.create(input=list(texts), model=model)
        vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
        return f"OpenAI {model}", vectors

    if list(texts) != SAMPLE_TEXTS:
        if list(texts) == [SAMPLE_QUERY]:
            return "offline semantic fixture", [OFFLINE_QUERY_VECTOR[:]]
        raise ValueError("The offline fixture only supports the built-in sample texts and query")
    return "offline semantic fixture", [vector[:] for vector in OFFLINE_VECTORS]


def store_embeddings(
    corpus: Sequence[TextChunk], vectors: Sequence[Sequence[float]]
) -> list[dict[str, Any]]:
    """Pair each source chunk and its metadata with its returned vector."""
    if len(corpus) != len(vectors):
        raise ValueError("Each source chunk must have exactly one embedding")
    return [
        {"text": chunk.text, "metadata": chunk.metadata, "embedding": list(vector)}
        for chunk, vector in zip(corpus, vectors)
    ]


def rank_chunks(
    query_embedding: Sequence[float], records: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Rank stored chunks by cosine similarity to a query embedding."""
    ranked = [
        {
            "score": cosine_similarity(query_embedding, record["embedding"]),
            "text": record["text"],
            "metadata": record["metadata"],
        }
        for record in records
    ]
    return sorted(ranked, key=lambda result: result["score"], reverse=True)


def build_report(
    provider: str,
    corpus: Sequence[TextChunk],
    vectors: Sequence[Sequence[float]],
    query: str = SAMPLE_QUERY,
    query_embedding: Sequence[float] | None = None,
) -> str:
    """Build the committed, human-readable demonstration report."""
    dimensions = [len(vector) for vector in vectors]
    if not dimensions or len(set(dimensions)) != 1:
        raise ValueError("Every sample text must produce a vector of the same length")

    stored_records = store_embeddings(corpus, vectors)
    if query_embedding is None:
        _, query_vectors = generate_embeddings([query])
        query_embedding = query_vectors[0]
    rankings = rank_chunks(query_embedding, stored_records)
    similar_score = cosine_similarity(vectors[0], vectors[1])
    unrelated_score = cosine_similarity(vectors[0], vectors[2])
    if similar_score <= unrelated_score:
        raise ValueError("The similar pair must score higher than the unrelated pair")

    lines = [
        "# Embedding Demonstration",
        "",
        f"Provider: `{provider}`",
        f"Chunks embedded: `{len(stored_records)}`",
        f"Vector dimension: `{dimensions[0]}`",
        f"All chunks have the same vector length: `{len(set(dimensions)) == 1}`",
        "",
        "## Stored chunks and vectors",
        "",
    ]
    for record in stored_records:
        trimmed_vector = record["embedding"][:5]
        lines.extend(
            [
                f"- **Text:** {record['text']}",
                f"  **Metadata:** `{json.dumps(record['metadata'], sort_keys=True)}`",
                f"  **Vector length:** `{len(record['embedding'])}`",
                f"  **Vector values (first 5):** `{json.dumps(trimmed_vector)}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Cosine similarity",
            "",
            f"- Similar pair (samples 1 and 2): `{similar_score:.6f}`",
            f"- Unrelated pair (samples 1 and 3): `{unrelated_score:.6f}`",
            f"- Similar pair scores higher: `{similar_score > unrelated_score}`",
            "",
            "## Query ranking",
            "",
            f"Query: **{query}**",
            "",
            "Metric: cosine similarity, because it compares the direction of embedding vectors and is less "
            "affected by their magnitude.",
            "",
        ]
    )
    for rank, result in enumerate(rankings, 1):
        lines.extend(
            [
                f"{rank}. **Score:** `{result['score']:.6f}`",
                f"   **Text:** {result['text']}",
                f"   **Metadata:** `{json.dumps(result['metadata'], sort_keys=True)}`",
            ]
        )
    lines.extend(
        [
            "",
            f"Most similar: **{rankings[0]['text']}** (`{rankings[0]['score']:.6f}`)",
            f"Least similar: **{rankings[-1]['text']}** (`{rankings[-1]['score']:.6f}`)",
            "",
            "## What vectors represent",
            "",
            "Embedding vectors are numeric representations of a text's meaning in a learned semantic space. "
            "They are not random IDs and they are not keyword counts; texts with related meaning tend to be "
            "near one another in that space, which is why cosine similarity is useful for semantic search.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_dimension(vectors: Sequence[Sequence[float]], expected: str | None) -> int:
    """Confirm one returned dimension and, when configured, the expected dimension."""
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError("The embeddings API returned inconsistent vector dimensions")
    dimension = dimensions.pop()
    if expected and dimension != int(expected):
        raise ValueError(f"Expected vector dimension {expected}, received {dimension}")
    return dimension


def main() -> None:
    client = None
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    embedding_model = os.getenv("EMBEDDING_MODEL")
    if api_key:
        from openai import OpenAI

        client_kwargs = {"api_key": api_key}
        api_base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("API_BASE_URL")
        if api_base_url:
            client_kwargs["base_url"] = api_base_url
        client = OpenAI(**client_kwargs)

    provider, vectors = generate_embeddings(
        SAMPLE_TEXTS, client=client, model=embedding_model
    )
    _, query_vectors = generate_embeddings(
        [SAMPLE_QUERY], client=client, model=embedding_model
    )
    validate_dimension(vectors, os.getenv("EMBEDDING_DIMENSION"))
    validate_dimension(query_vectors, os.getenv("EMBEDDING_DIMENSION"))
    report = build_report(provider, SAMPLE_CORPUS, vectors, query_embedding=query_vectors[0])
    output_path = Path(os.getenv("EMBEDDING_OUTPUT", "outputs/embedding_demo.md"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()