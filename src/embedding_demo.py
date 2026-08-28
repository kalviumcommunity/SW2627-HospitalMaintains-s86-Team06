"""Generate and compare a few text embeddings.

With an OpenAI client this calls the configured embeddings model. Without one,
the small deterministic semantic fixture keeps the educational demo runnable
offline; it is not a replacement for a production embedding model.
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Sequence


SAMPLE_TEXTS = [
    "The patient should take the prescribed medication with water.",
    "Patients need to use their recommended medicine with water.",
    "The help desk can reset an employee password.",
]

# Deterministic semantic fixture for environments without API credentials.
OFFLINE_VECTORS = [
    [0.91, 0.08, 0.02, 0.12, 0.04, 0.03, 0.01, 0.02],
    [0.89, 0.10, 0.03, 0.11, 0.05, 0.02, 0.01, 0.03],
    [0.04, 0.02, 0.91, 0.03, 0.08, 0.02, 0.01, 0.05],
]


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
    texts: Sequence[str], client: Any | None = None, model: str = "text-embedding-3-small"
) -> tuple[str, list[list[float]]]:
    """Generate vectors through OpenAI when available, otherwise use the offline fixture."""
    if client is not None:
        response = client.embeddings.create(input=list(texts), model=model)
        vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
        return f"OpenAI {model}", vectors

    if list(texts) != SAMPLE_TEXTS:
        raise ValueError("The offline fixture only supports the built-in sample texts")
    return "offline semantic fixture", [vector[:] for vector in OFFLINE_VECTORS]


def build_report(provider: str, texts: Sequence[str], vectors: Sequence[Sequence[float]]) -> str:
    """Build the committed, human-readable demonstration report."""
    dimensions = [len(vector) for vector in vectors]
    if not dimensions or len(set(dimensions)) != 1:
        raise ValueError("Every sample text must produce a vector of the same length")

    similar_score = cosine_similarity(vectors[0], vectors[1])
    unrelated_score = cosine_similarity(vectors[0], vectors[2])
    if similar_score <= unrelated_score:
        raise ValueError("The similar pair must score higher than the unrelated pair")

    lines = [
        "# Embedding Demonstration",
        "",
        f"Provider: `{provider}`",
        f"Vector dimension: `{dimensions[0]}`",
        f"All {len(vectors)} sample texts have the same vector length: `{len(set(dimensions)) == 1}`",
        "",
        "## Sample vectors",
        "",
    ]
    for text, vector in zip(texts, vectors):
        lines.extend([f"- **Text:** {text}", f"  **Vector:** `{json.dumps(list(vector))}`"])
    lines.extend(
        [
            "",
            "## Cosine similarity",
            "",
            f"- Similar pair (samples 1 and 2): `{similar_score:.6f}`",
            f"- Unrelated pair (samples 1 and 3): `{unrelated_score:.6f}`",
            f"- Similar pair scores higher: `{similar_score > unrelated_score}`",
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


def main() -> None:
    client = None
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI

        client = OpenAI()

    provider, vectors = generate_embeddings(SAMPLE_TEXTS, client=client)
    report = build_report(provider, SAMPLE_TEXTS, vectors)
    output_path = Path(os.getenv("EMBEDDING_OUTPUT", "outputs/embedding_demo.md"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()