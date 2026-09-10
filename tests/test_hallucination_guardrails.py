from src.hallucination_guardrails import SAFE_REFUSAL, evaluate_retrieval
from src.embedding_demo import SAMPLE_QUERY
from src.rag_pipeline import RetrievedSource, run_rag_pipeline


def test_empty_retrieval_is_rejected():
    decision = evaluate_retrieval([])

    assert not decision.allowed
    assert decision.qualifying_sources == 0
    assert decision.reason == "retrieval returned no sources"


def test_low_similarity_results_are_rejected():
    sources = [RetrievedSource("Unrelated text", {"source_document": "it.pdf"}, 0.2)]

    decision = evaluate_retrieval(sources, minimum_score=0.75)

    assert not decision.allowed
    assert "0.75 relevance threshold" in decision.reason


def test_confident_retrieval_is_allowed():
    sources = [
        RetrievedSource("Relevant text", {"source_document": "guide.pdf"}, 0.91),
        RetrievedSource("Another relevant passage", {"source_document": "guide.pdf"}, 0.82),
    ]

    decision = evaluate_retrieval(sources, minimum_score=0.75, minimum_chunks=2)

    assert decision.allowed
    assert decision.qualifying_sources == 2


def test_pipeline_refuses_weak_sample_query_and_answers_strong_query():
    weak_response = run_rag_pipeline(SAMPLE_QUERY, k=2, minimum_score=1.0)
    strong_response = run_rag_pipeline(SAMPLE_QUERY, k=2)

    assert weak_response.refused
    assert weak_response.answer == SAFE_REFUSAL
    assert weak_response.refusal_reason is not None
    assert not strong_response.refused
    assert "[1]" in strong_response.answer