import pytest
from src.text_cleaner import clean_text, clean, clean_document, clean_corpus


def test_unicode_nfkc_normalization():
    # Test NFKC normalization fixes full-width characters and unicode representation
    raw_text = "Clinicalâ€™s Guidelines \uff11\uff12\uff13"
    cleaned = clean_text(raw_text)
    assert "123" in cleaned
    assert isinstance(cleaned, str)


def test_line_ending_normalization():
    raw_text = "Line 1\r\nLine 2\rLine 3\nLine 4"
    cleaned = clean_text(raw_text)
    assert "\r" not in cleaned
    assert cleaned == "Line 1\nLine 2\nLine 3\nLine 4"


def test_footer_boilerplate_removal():
    raw_text = "Section 1: Overview\nPage 3 of 12\nSection 2: Protocol\nPage 4"
    cleaned = clean_text(raw_text)
    assert "Page 3 of 12" not in cleaned
    assert "Page 4" not in cleaned
    assert "Section 1: Overview" in cleaned
    assert "Section 2: Protocol" in cleaned


def test_whitespace_collapsing():
    raw_text = "This   is   a    test\t\twith   extra   spaces."
    cleaned = clean_text(raw_text)
    assert cleaned == "This is a test with extra spaces."


def test_blank_line_collapsing():
    raw_text = "Header\n\n\n\n\nBody content\n\n\nFooter content"
    cleaned = clean_text(raw_text)
    assert "\n\n\n" not in cleaned
    assert cleaned == "Header\n\nBody content\n\nFooter content"


def test_clean_alias():
    # Test clean alias behaves identically to clean_text
    raw = "Test   Page 1 of 5\r\nText"
    assert clean(raw) == clean_text(raw)


def test_clean_document():
    raw = "Protocol   Header\r\nPage 1 of 2\n\n\nContent line."
    doc = {
        "source": "sample_protocol.txt",
        "format": ".txt",
        "text": raw,
        "char_count": len(raw),
    }
    cleaned_doc = clean_document(doc)

    assert cleaned_doc["source"] == "sample_protocol.txt"
    assert cleaned_doc["before_char_count"] == len(raw)
    assert "Page 1 of 2" not in cleaned_doc["text"]
    assert cleaned_doc["char_count"] == len(cleaned_doc["text"])
    assert "\r" not in cleaned_doc["text"]


def test_clean_corpus():
    docs = [
        {"source": "doc1.txt", "text": "Doc 1   text  Page 1 of 10"},
        {"source": "doc2.txt", "text": "Doc 2   text\r\n\n\n\nMore text"},
    ]
    cleaned_docs = clean_corpus(docs)

    assert len(cleaned_docs) == 2
    assert "Page 1 of 10" not in cleaned_docs[0]["text"]
    assert "\n\n\n" not in cleaned_docs[1]["text"]


def test_preserves_structure_and_meaning():
    raw_text = """# CARDIOLOGY CLINICAL TRIAGE PROTOCOL

## 1. Acute Coronary Syndrome (ACS) Triage
- **Initial Assessment**: Complete 12-lead ECG within 10 minutes of patient arrival.
- **Biomarker Protocol**: Draw High-Sensitivity Troponin I at arrival (0h) and 3h post-presentation.
- **Medication Administration**: Administer Aspirin 300mg chewable immediately unless contraindicated.
"""
    cleaned = clean_text(raw_text)

    # Ensure punctuation, numbers, markdown symbols, and headings are preserved
    assert "# CARDIOLOGY CLINICAL TRIAGE PROTOCOL" in cleaned
    assert "## 1. Acute Coronary Syndrome (ACS) Triage" in cleaned
    assert "- **Initial Assessment**:" in cleaned
    assert "Aspirin 300mg" in cleaned


def test_empty_and_non_string():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text(12345) == ""
