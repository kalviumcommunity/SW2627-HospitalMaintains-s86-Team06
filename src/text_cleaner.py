import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Any, Union

try:
    from src.document_loader import load_directory, load_document
except ImportError:
    from document_loader import load_directory, load_document


def clean_text(text: str) -> str:
    """
    Cleans raw extracted document text to produce consistent, high-quality text for RAG indexing.

    Cleaning steps:
    1. Unicode normalization (NFKC) to resolve encoding artifacts, accents, and mojibake.
    2. Normalize line endings (CRLF/CR -> LF).
    3. Strip boilerplate page footers (e.g., 'Page 3 of 12').
    4. Collapse consecutive spaces and tabs into a single space on each line.
    5. Collapse runaway blank lines (3 or more newlines into double newlines).
    6. Strip leading and trailing whitespace.

    Args:
        text: Raw text extracted from documents.

    Returns:
        Cleaned, normalized text string.
    """
    if not isinstance(text, str):
        return ""

    # 1. Fix encoding artifacts via NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Remove footer & header boilerplate like "Page X of Y" or "Page X"
    text = re.sub(r"(?i)Page \d+ of \d+", "", text)
    text = re.sub(r"(?i)Page \d+", "", text)

    # 4. Collapse consecutive horizontal whitespace (spaces & tabs)
    text = re.sub(r"[ \t]+", " ", text)

    # 5. Collapse runaway blank lines (3+ consecutive newlines -> 2 newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 6. Trim leading and trailing whitespace
    return text.strip()


# Alias clean to clean_text for direct compatibility with concept documentation
clean = clean_text


def clean_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cleans the 'text' field of a document metadata dictionary and updates metrics.

    Args:
        doc: Document dictionary containing at least 'text' and 'source'.

    Returns:
        New or updated document dictionary with cleaned text and character counts.
    """
    cleaned_doc = doc.copy()
    before_text = doc.get("text", "")
    after_text = clean_text(before_text)

    cleaned_doc["text"] = after_text
    cleaned_doc["before_char_count"] = len(before_text)
    cleaned_doc["char_count"] = len(after_text)
    cleaned_doc["sample"] = after_text[:60].replace("\n", " ")
    return cleaned_doc


def clean_corpus(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Applies the text cleaning pipeline uniformly across a list of documents.

    Args:
        docs: List of document dictionaries.

    Returns:
        List of document dictionaries with cleaned text.
    """
    cleaned_docs = []
    for d in docs:
        cleaned_docs.append(clean_document(d))
    return cleaned_docs


def main():
    print("\n=======================================================")
    print(" [TEXT CLEANING] DOCUMENT EXTRACTION & CLEANING PIPELINE")
    print("=======================================================\n")

    data_dir = Path("data")
    if not data_dir.exists():
        print(f"Data directory '{data_dir}' not found.")
        return

    print(f"Loading raw documents from: {data_dir.resolve()}...\n")
    docs = load_directory(data_dir)

    print(f"\n-------------------------------------------------------")
    print(f" Running Cleaning Pipeline Across {len(docs)} Document(s)")
    print(f"-------------------------------------------------------\n")

    cleaned_docs = clean_corpus(docs)

    for idx, (raw_doc, cleaned_doc) in enumerate(zip(docs, cleaned_docs), 1):
        before = raw_doc["text"]
        after = cleaned_doc["text"]
        print(f"[{idx}] {cleaned_doc['source']}: {len(before)} -> {len(after)} chars")
        print(f"  BEFORE snippet: {before[:100]!r}")
        print(f"  AFTER  snippet: {after[:100]!r}")
        print("-" * 55)

    total_before = sum(d["char_count"] for d in docs)
    total_after = sum(d["char_count"] for d in cleaned_docs)
    print(f"\nPipeline Summary: {total_before:,} chars -> {total_after:,} chars cleaned.")


if __name__ == "__main__":
    main()
