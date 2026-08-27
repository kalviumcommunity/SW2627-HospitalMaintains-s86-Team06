import pytest
from pathlib import Path
from src.document_loader import (
    load_text,
    load_document,
    load_directory,
    SUPPORTED_EXTENSIONS
)

DATA_DIR = Path("data")


def test_load_text_txt():
    txt_path = DATA_DIR / "policy.txt"
    text = load_text(txt_path)
    assert isinstance(text, str)
    assert "HOSPITAL CLINICAL PROTOCOL" in text
    assert len(text) > 0


def test_load_text_md():
    md_path = DATA_DIR / "protocol_cardiology.md"
    text = load_text(md_path)
    assert isinstance(text, str)
    assert "CARDIOLOGY CLINICAL TRIAGE PROTOCOL" in text
    assert len(text) > 0


def test_load_text_html():
    html_path = DATA_DIR / "drug_interactions.html"
    text = load_text(html_path)
    assert isinstance(text, str)
    assert "PHARMACY DRUG INTERACTION GUIDELINES" in text
    assert "Warfarin" in text
    assert "<html>" not in text
    assert "<table>" not in text


def test_load_text_pdf():
    pdf_path = DATA_DIR / "emergency_guidelines.pdf"
    text = load_text(pdf_path)
    assert isinstance(text, str)
    assert "EMERGENCY RESPONSE PROTOCOL" in text


def test_load_text_unsupported():
    zip_path = DATA_DIR / "unsupported_archive.zip"
    with pytest.raises(ValueError) as exc_info:
        load_text(zip_path)
    assert "unsupported" in str(exc_info.value)


def test_load_text_nonexistent():
    fake_path = DATA_DIR / "non_existent_file.txt"
    with pytest.raises(FileNotFoundError):
        load_text(fake_path)


def test_load_document_metadata():
    txt_path = DATA_DIR / "policy.txt"
    doc = load_document(txt_path)
    assert doc["source"] == "policy.txt"
    assert doc["format"] == ".txt"
    assert isinstance(doc["char_count"], int)
    assert doc["char_count"] == len(doc["text"])
    assert len(doc["sample"]) <= 60


def test_load_directory_graceful_skipping():
    docs = load_directory(DATA_DIR)
    assert isinstance(docs, list)
    assert len(docs) >= 4

    sources = [d["source"] for d in docs]
    assert "policy.txt" in sources
    assert "protocol_cardiology.md" in sources
    assert "drug_interactions.html" in sources
    assert "emergency_guidelines.pdf" in sources

    # Corrupt and unsupported files should be skipped and not present in loaded docs
    assert "corrupt_doc.pdf" not in sources
    assert "unsupported_archive.zip" not in sources
