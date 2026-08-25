import pytest
from src.document_chunker import (
    fixed_chunks,
    paragraph_chunks,
    sentence_chunks,
    compare_chunk_strategies,
    chunk_document,
    chunk_corpus,
)


def test_fixed_chunks_basic_and_overlap():
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    # size = 10, overlap = 2 -> step = 8
    chunks = fixed_chunks(text, size=10, overlap=2)
    assert len(chunks) > 1
    assert chunks[0] == "ABCDEFGHIJKLMNOP"[:10]  # "ABCDEFGHIJ"
    # check overlap between first and second chunk
    assert chunks[0][-2:] == chunks[1][:2]


def test_fixed_chunks_short_text():
    text = "Short text"
    chunks = fixed_chunks(text, size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == "Short text"


def test_fixed_chunks_invalid_overlap():
    with pytest.raises(ValueError):
        fixed_chunks("Sample text", size=50, overlap=50)


def test_paragraph_chunks_basic():
    text = "Paragraph 1: Overview of clinical policy.\n\nParagraph 2: Detailed dosage and procedures.\n\nParagraph 3: Emergency contact information."
    chunks = paragraph_chunks(text)
    assert len(chunks) == 3
    assert chunks[0] == "Paragraph 1: Overview of clinical policy."
    assert chunks[1] == "Paragraph 2: Detailed dosage and procedures."
    assert chunks[2] == "Paragraph 3: Emergency contact information."


def test_paragraph_chunks_multiple_blank_lines():
    text = "Header\n\n\n\nBody content\n\nFooter"
    chunks = paragraph_chunks(text)
    assert len(chunks) == 3
    assert chunks == ["Header", "Body content", "Footer"]


def test_sentence_chunks_basic():
    text = "First sentence here. Second sentence follows. Third sentence ends it."
    chunks = sentence_chunks(text, max_chunk_size=45)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= 45


def test_compare_chunk_strategies():
    text = "Paragraph 1 content here.\n\nParagraph 2 content here.\n\nParagraph 3 content here."
    results = compare_chunk_strategies(text, doc_name="test_doc.md")

    assert results["doc_name"] == "test_doc.md"
    assert "fixed" in results["strategies"]
    assert "paragraph" in results["strategies"]
    assert "sentence" in results["strategies"]

    para_stats = results["strategies"]["paragraph"]
    assert para_stats["count"] == 3
    assert para_stats["avg_size"] > 0
    assert para_stats["min_size"] <= para_stats["max_size"]
    assert isinstance(para_stats["sample"], str)


def test_chunk_document_metadata():
    doc = {
        "source": "protocol_cardiology.md",
        "text": "Header paragraph.\n\nBody paragraph with clinical instructions.",
    }
    chunks = chunk_document(doc, strategy="paragraph")

    assert len(chunks) == 2
    first_chunk = chunks[0]
    assert first_chunk["chunk_id"] == "protocol_cardiology.md_chunk_1"
    assert first_chunk["source"] == "protocol_cardiology.md"
    assert first_chunk["chunk_index"] == 1
    assert first_chunk["strategy"] == "paragraph"
    assert first_chunk["char_count"] == len(first_chunk["text"])


def test_chunk_document_invalid_strategy():
    doc = {"source": "test.txt", "text": "Some content."}
    with pytest.raises(ValueError):
        chunk_document(doc, strategy="invalid_strategy_name")


def test_chunk_corpus_batch():
    docs = [
        {"source": "doc1.txt", "text": "Doc 1 P1\n\nDoc 1 P2"},
        {"source": "doc2.txt", "text": "Doc 2 P1\n\nDoc 2 P2\n\nDoc 2 P3"},
    ]
    all_chunks = chunk_corpus(docs, strategy="paragraph")

    assert len(all_chunks) == 5
    sources = [c["source"] for c in all_chunks]
    assert sources.count("doc1.txt") == 2
    assert sources.count("doc2.txt") == 3


def test_empty_and_whitespace_handling():
    assert fixed_chunks("") == []
    assert paragraph_chunks("   \n\n   ") == []
    assert sentence_chunks("") == []
    assert chunk_corpus([]) == []
