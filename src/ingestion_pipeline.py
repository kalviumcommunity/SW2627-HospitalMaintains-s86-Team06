"""End-to-end corpus ingestion with auditable completeness reporting."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.document_chunker import chunk_corpus
from src.document_loader import SUPPORTED_EXTENSIONS, load_document
from src.text_cleaner import clean_corpus


def run_ingestion(
    data_dir: Path, output_dir: Path, strategy: str = "paragraph"
) -> Dict[str, Any]:
    """Load, clean, chunk, validate, and persist a complete corpus run."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_paths = sorted(path for path in data_dir.rglob("*") if path.is_file())
    documents: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    for path in source_paths:
        try:
            documents.append(load_document(path))
        except Exception as error:
            failures.append(
                {
                    "source": path.name,
                    "path": str(path),
                    "reason": str(error),
                    "category": "unsupported" if path.suffix.lower() not in SUPPORTED_EXTENSIONS else "load_error",
                }
            )

    cleaned_documents = clean_corpus(documents)
    chunks = chunk_corpus(cleaned_documents, strategy=strategy)
    summary = {
        "data_directory": str(data_dir.resolve()),
        "strategy": strategy,
        "total_source_documents": len(source_paths),
        "successfully_ingested_documents": len(documents),
        "total_chunks_created": len(chunks),
        "failure_count": len(failures),
        "skipped_files": failures,
        "completeness_check": {
            "passed": len(source_paths) == len(documents) + len(failures),
            "source_documents": len(source_paths),
            "ingested_plus_failures": len(documents) + len(failures),
        },
    }
    if not summary["completeness_check"]["passed"]:
        raise RuntimeError("Completeness check failed: source documents do not reconcile")

    samples = [
        {
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "strategy": chunk["strategy"],
            "char_count": chunk["char_count"],
            "text": chunk["text"],
        }
        for chunk in chunks[:3]
    ]
    (output_dir / "ingestion_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "sample_chunks.json").write_text(
        json.dumps(samples, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete corpus ingestion")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    summary = run_ingestion(args.data_dir, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()