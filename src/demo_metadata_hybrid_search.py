"""Demonstration script for Metadata Filtering and Hybrid Search.

Executes and compares:
1. Unfiltered vector search across the entire corpus.
2. Metadata-filtered vector search scoped to section = "Account access".
3. Hybrid search combining semantic vector scores with lexical keyword scoring.
Generates an output report saved to outputs/metadata_hybrid_search_demo.md.
"""

from __future__ import annotations

from pathlib import Path
from src.metadata_hybrid_search import (
    build_metadata_hybrid_report,
    hybrid_rank,
    retrieve,
    show_results,
)


def main() -> None:
    query = "What are the password reset steps?"
    filter_dict = {"section": "Account access"}
    keywords = ["password", "reset"]

    print("=" * 60)
    print("DEMO: Metadata Filtering and Hybrid Search Retrieval")
    print("=" * 60)
    print(f"Query: '{query}'\n")

    # 1. Unfiltered Retrieval
    unfiltered = retrieve(query, k=3)
    show_results("--- UNFILTERED VECTOR RETRIEVAL ---", unfiltered)

    # 2. Metadata-Filtered Retrieval
    filtered = retrieve(query, k=3, metadata_filter=filter_dict)
    show_results("--- METADATA-FILTERED RETRIEVAL (section = Account access) ---", filtered)

    # 3. Hybrid Lexical-Semantic Search
    hybrid = hybrid_rank(filtered, keywords=keywords)
    show_results("--- HYBRID FILTERED RETRIEVAL (keywords = password, reset) ---", hybrid)

    # 4. Generate & Save Report
    report = build_metadata_hybrid_report(
        query=query,
        unfiltered_results=unfiltered,
        filtered_results=filtered,
        hybrid_results=hybrid,
        metadata_filter=filter_dict,
        keywords=keywords,
    )

    output_path = Path("outputs/metadata_hybrid_search_demo.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Report generated successfully and saved to {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
