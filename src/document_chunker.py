import re
from pathlib import Path
from typing import Dict, List, Any, Union

try:
    from src.document_loader import load_directory
    from src.text_cleaner import clean_corpus, clean_text
except ImportError:
    from document_loader import load_directory
    from text_cleaner import clean_corpus, clean_text


def fixed_chunks(text: str, size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits text into fixed-size character windows with a sliding overlap.

    Args:
        text: Input document text.
        size: Target character length per chunk (default 500).
        overlap: Character overlap between consecutive chunks (default 50).

    Returns:
        List of text chunk strings.
    """
    step = size - overlap
    if step <= 0:
        raise ValueError("Chunk size must be greater than overlap.")

    if not text or not text.strip():
        return []

    text = text.strip()
    if len(text) <= size:
        return [text]

    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i : i + size]
        chunks.append(chunk)
        if i + size >= len(text):
            break
        i += step

    return chunks


def paragraph_chunks(text: str) -> List[str]:
    """
    Splits text along double-newline paragraph boundaries (\n\n).

    Args:
        text: Input document text.

    Returns:
        List of non-empty paragraph chunk strings.
    """
    if not text or not text.strip():
        return []

    paragraphs = text.split("\n\n")
    cleaned_paragraphs = [p.strip() for p in paragraphs if p and p.strip()]
    return cleaned_paragraphs


def sentence_chunks(text: str, max_chunk_size: int = 500) -> List[str]:
    """
    Splits text into sentence units and groups them into chunks without exceeding max_chunk_size.

    Args:
        text: Input document text.
        max_chunk_size: Maximum character length for a grouped chunk.

    Returns:
        List of sentence-grouped text chunk strings.
    """
    if not text or not text.strip():
        return []

    # Split by common sentence terminators (. ! ?) followed by whitespace
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if current_chunk and (current_length + sentence_len + 1 > max_chunk_size):
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_len
        else:
            current_chunk.append(sentence)
            current_length += sentence_len + (1 if current_chunk else 0)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def compare_chunk_strategies(text: str, doc_name: str = "") -> Dict[str, Any]:
    """
    Compares fixed-size, paragraph, and sentence chunking strategies on a document.

    Args:
        text: Cleaned text to chunk.
        doc_name: Optional identifier for reporting.

    Returns:
        Dictionary containing comparative statistics and sample boundaries for each strategy.
    """
    strategies = {
        "fixed": fixed_chunks(text, size=500, overlap=50),
        "paragraph": paragraph_chunks(text),
        "sentence": sentence_chunks(text, max_chunk_size=500),
    }

    results = {"doc_name": doc_name, "strategies": {}}

    for name, chunks in strategies.items():
        if not chunks:
            results["strategies"][name] = {
                "count": 0,
                "avg_size": 0,
                "min_size": 0,
                "max_size": 0,
                "sample": "",
            }
            continue

        sizes = [len(c) for c in chunks]
        results["strategies"][name] = {
            "count": len(chunks),
            "avg_size": sum(sizes) // len(sizes),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "sample": chunks[0][:100],
        }

    return results


def chunk_document(
    doc: Dict[str, Any], strategy: str = "paragraph", **kwargs
) -> List[Dict[str, Any]]:
    """
    Splits a document dictionary's text into structured chunk dictionaries with metadata.

    Args:
        doc: Document dictionary containing 'text' and 'source'.
        strategy: Strategy name ('fixed', 'paragraph', 'sentence').
        **kwargs: Additional parameters passed to the chunking function (size, overlap, max_chunk_size).

    Returns:
        List of chunk dictionaries with metadata.
    """
    text = doc.get("text", "")
    source = doc.get("source", "unknown")

    if strategy == "fixed":
        raw_chunks = fixed_chunks(text, **kwargs)
    elif strategy == "paragraph":
        raw_chunks = paragraph_chunks(text)
    elif strategy == "sentence":
        raw_chunks = sentence_chunks(text, **kwargs)
    else:
        raise ValueError(f"Unsupported chunking strategy: {strategy}")

    chunk_objects = []
    for idx, chunk_text in enumerate(raw_chunks, 1):
        chunk_objects.append(
            {
                "chunk_id": f"{source}_chunk_{idx}",
                "source": source,
                "chunk_index": idx,
                "text": chunk_text,
                "char_count": len(chunk_text),
                "strategy": strategy,
            }
        )

    return chunk_objects


def chunk_corpus(
    docs: List[Dict[str, Any]], strategy: str = "paragraph", **kwargs
) -> List[Dict[str, Any]]:
    """
    Applies document chunking across a full corpus of documents.

    Args:
        docs: List of document dictionaries.
        strategy: Chunking strategy to apply.
        **kwargs: Strategy parameters.

    Returns:
        Flattened list of chunk dictionaries for all documents in the corpus.
    """
    all_chunks = []
    for doc in docs:
        doc_chunks = chunk_document(doc, strategy=strategy, **kwargs)
        all_chunks.extend(doc_chunks)
    return all_chunks


def main():
    print("\n=======================================================")
    print(" [DOCUMENT CHUNKING] STRATEGY COMPARISON & CORPUS PARTITIONING")
    print("=======================================================\n")

    data_dir = Path("data")
    if not data_dir.exists():
        print(f"Data directory '{data_dir}' not found.")
        return

    print(f"Loading and cleaning documents from: {data_dir.resolve()}...\n")
    raw_docs = load_directory(data_dir)
    docs = clean_corpus(raw_docs)

    print(f"\n-------------------------------------------------------")
    print(f" Strategy Comparison across {len(docs)} Document(s)")
    print(f"-------------------------------------------------------\n")

    for doc in docs:
        comparison = compare_chunk_strategies(doc["text"], doc_name=doc["source"])
        print(f"[DOC] Document: {doc['source']} ({doc['char_count']} chars)")
        print(f"  {'Strategy':<12} | {'Chunks':<7} | {'Avg Size':<9} | {'Min-Max Size':<12}")
        print(f"  {'-'*12}-+-{'-'*7}-+-{'-'*9}-+-{'-'*12}")

        for strat_name, stats in comparison["strategies"].items():
            min_max = f"{stats['min_size']}-{stats['max_size']} chars"
            print(
                f"  {strat_name:<12} | {stats['count']:<7} | {stats['avg_size']:<5} chars | {min_max:<12}"
            )
            print(f"    Sample boundary: {stats['sample']!r}...")
        print("-" * 55)

    print("\n-------------------------------------------------------")
    print(" Corpus Partitioning (Selected Strategy: Paragraph)")
    print("-------------------------------------------------------")
    corpus_chunks = chunk_corpus(docs, strategy="paragraph")

    print(f"Total Chunks Generated: {len(corpus_chunks)}")
    avg_chunk_size = sum(c["char_count"] for c in corpus_chunks) // max(len(corpus_chunks), 1)
    print(f"Average Chunk Size   : {avg_chunk_size} chars\n")

    print("Sample Chunks Generated for RAG Retrieval:")
    for chunk in corpus_chunks[:3]:
        print(f"  [{chunk['chunk_id']}] ({chunk['char_count']} chars)")
        print(f"  Text snippet: {chunk['text'][:120]!r}")
        print("  " + "." * 40)

    print("\n[INFO] Strategy Justification:")
    print(
        "  - Paragraph chunking is selected for structured clinical protocols and guidelines."
    )
    print(
        "  - It preserves complete semantic units (triage steps, dosages, contraindications)."
    )
    print(
        "  - Fixed-size chunking risks splitting a critical drug dosage or emergency step across boundaries."
    )


if __name__ == "__main__":
    main()
