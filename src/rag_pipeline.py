"""Composable query-to-answer RAG pipeline.

The default path uses the deterministic embedding fixture from
``embedding_demo`` so the complete flow can be demonstrated without API
credentials. Production callers can inject an OpenAI-compatible client for
embedding and chat generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from src.context_assembly import build_augmented_prompt
from src.embedding_demo import (
    SAMPLE_CORPUS,
    SAMPLE_QUERY,
    TextChunk,
    embed_query,
    generate_embeddings,
    store_embeddings,
    top_k_similarity_search,
)


@dataclass(frozen=True)
class RetrievedSource:
    """One retrieved chunk that can be returned as an answer citation."""

    text: str
    metadata: dict[str, Any]
    score: float


@dataclass(frozen=True)
class RagResponse:
    """Final answer and the sources used to ground it."""

    query: str
    answer: str
    sources: list[RetrievedSource]
    embedding_provider: str


def embed_query_stage(
    query: str, embedding_client: Any | None = None, embedding_model: str | None = None
) -> tuple[str, list[float]]:
    """Embed the user query in the same vector space as indexed chunks."""
    if not query.strip():
        raise ValueError("query must not be empty")
    return embed_query(query, client=embedding_client, model=embedding_model)


def retrieve_stage(
    query_embedding: Sequence[float],
    indexed_records: Sequence[dict[str, Any]],
    k: int = 3,
) -> list[RetrievedSource]:
    """Retrieve the top-k chunks ranked by cosine similarity."""
    return [
        RetrievedSource(
            text=result["text"],
            metadata=dict(result["metadata"]),
            score=float(result["score"]),
        )
        for result in top_k_similarity_search(query_embedding, indexed_records, k=k)
    ]


def assemble_context(sources: Sequence[RetrievedSource]) -> str:
    """Turn retrieved chunks into numbered, citation-friendly context."""
    return "\n\n".join(
        f"[Source {index}] {source.text}"
        for index, source in enumerate(sources, start=1)
    )


def generate_answer(
    query: str,
    context: str,
    sources: Sequence[RetrievedSource],
    chat_client: Any | None = None,
    chat_model: str | None = None,
    model_token_budget: int = 256,
    reserved_answer_tokens: int = 64,
) -> str:
    """Generate a grounded answer, or decline when retrieval found nothing."""
    if not sources or not context.strip():
        return "I could not find supporting information in the indexed documents."

    augmented_prompt = build_augmented_prompt(
        query,
        sources,
        model_token_budget=model_token_budget,
        reserved_answer_tokens=reserved_answer_tokens,
    )

    if chat_client is not None:
        if not chat_model:
            raise ValueError("A chat model is required when using a chat client")
        response = chat_client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": augmented_prompt.system_message},
                {"role": "user", "content": augmented_prompt.user_message},
            ],
        )
        return response.choices[0].message.content

    # Deterministic offline generation keeps the end-to-end demonstration runnable.
    selected_sources = sources[: augmented_prompt.included_sources]
    source_labels = ", ".join(f"[{index}]" for index in range(1, len(selected_sources) + 1))
    evidence = " ".join(source.text for source in selected_sources)
    return f"Based on the indexed documents, {evidence} ({source_labels})."


def build_index_records(
    corpus: Sequence[TextChunk] = SAMPLE_CORPUS,
    embedding_client: Any | None = None,
    embedding_model: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Embed corpus chunks once and return records ready for retrieval."""
    provider, vectors = generate_embeddings(
        [chunk.text for chunk in corpus],
        client=embedding_client,
        model=embedding_model,
    )
    return provider, store_embeddings(corpus, vectors)


def run_rag_pipeline(
    query: str = SAMPLE_QUERY,
    indexed_records: Sequence[dict[str, Any]] | None = None,
    embedding_client: Any | None = None,
    embedding_model: str | None = None,
    chat_client: Any | None = None,
    chat_model: str | None = None,
    k: int = 2,
) -> RagResponse:
    """Run embed -> retrieve -> assemble -> generate and return sources."""
    if indexed_records is None:
        provider, indexed_records = build_index_records(
            embedding_client=embedding_client,
            embedding_model=embedding_model,
        )
    else:
        provider = "pre-built index"

    embedding_provider, query_embedding = embed_query_stage(
        query,
        embedding_client=embedding_client,
        embedding_model=embedding_model,
    )
    sources = retrieve_stage(query_embedding, indexed_records, k=k)
    context = assemble_context(sources)
    answer = generate_answer(
        query,
        context,
        sources,
        chat_client=chat_client,
        chat_model=chat_model,
    )
    return RagResponse(
        query=query,
        answer=answer,
        sources=sources,
        embedding_provider=provider or embedding_provider,
    )
