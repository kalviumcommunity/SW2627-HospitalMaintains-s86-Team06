from src.rag_pipeline import RetrievedSource, generate_answer
from src.source_citations import build_citations, citation_mapping, verify_citation


def test_citations_map_markers_to_metadata_and_original_text():
    source = RetrievedSource(
        "Take the medication with water.",
        {"source_document": "medication-guideline.pdf", "chunk_index": 1, "page": 1},
        0.99,
    )

    citations = build_citations([source])
    mapping = citation_mapping(citations)

    assert citations[0].marker == "[1]"
    assert citations[0].source_document == "medication-guideline.pdf"
    assert mapping[0]["metadata"]["page"] == 1
    assert mapping[0]["retrieved_text"] == source.text
    assert verify_citation(citations[0], source.text)


def test_citation_verification_rejects_changed_text():
    source = RetrievedSource("Original chunk", {"source_document": "guide.pdf"}, 0.8)
    citation = build_citations([source])[0]

    assert not verify_citation(citation, "Changed chunk")


def test_answer_cites_only_real_sources_and_falls_back_without_sources():
    source = RetrievedSource(
        "The patient should take the prescribed medication with water.",
        {"source_document": "medication-guideline.pdf", "section": "Dosage"},
        0.99,
    )

    answer = generate_answer("How should it be taken?", source.text, [source])
    fallback = generate_answer("Unknown question", "", [])

    assert "[1]" in answer
    assert "[2]" not in answer
    assert "could not find supporting information" in fallback