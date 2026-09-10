from src.embedding_demo import SAMPLE_QUERY, SAMPLE_CORPUS, generate_embeddings, store_embeddings
from src.rag_pipeline import (
    RetrievedSource,
    assemble_context,
    generate_answer,
    run_rag_pipeline,
)


def test_pipeline_returns_grounded_answer_and_sources():
    response = run_rag_pipeline(SAMPLE_QUERY, k=2)

    assert response.embedding_provider == "offline semantic fixture"
    assert "prescribed medication with water" in response.answer
    assert len(response.sources) == 2
    assert response.sources[0].metadata["source_document"] == "medication-guideline.pdf"
    assert response.sources[0].score >= response.sources[1].score


def test_context_assembly_preserves_source_numbers():
    sources = [
        RetrievedSource("first passage", {"source_document": "one.pdf"}, 0.9),
        RetrievedSource("second passage", {"source_document": "two.pdf"}, 0.8),
    ]

    context = assemble_context(sources)

    assert context == "[Source 1] first passage\n\n[Source 2] second passage"


def test_generation_declines_when_retrieval_returns_nothing():
    answer = generate_answer("unknown question", "", [])

    assert answer == "I could not find supporting information in the indexed documents."


def test_pipeline_accepts_a_prebuilt_index():
    _, vectors = generate_embeddings([chunk.text for chunk in SAMPLE_CORPUS])
    records = store_embeddings(SAMPLE_CORPUS, vectors)

    response = run_rag_pipeline(SAMPLE_QUERY, indexed_records=records, k=1)

    assert response.embedding_provider == "pre-built index"
    assert len(response.sources) == 1