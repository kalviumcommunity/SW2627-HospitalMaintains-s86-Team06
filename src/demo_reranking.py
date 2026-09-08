"""Demonstration script for RAG Candidate Retrieval and Re-ranking.

Demonstrates:
1. Stage 1: Retrieving a candidate pool larger than final context (k=10).
2. Stage 2: Re-ranking candidates using query-chunk prompt scoring.
3. Comparing before-and-after candidate ordering.
4. Explaining cost and latency trade-offs.
5. Saving a Markdown report to outputs/reranking_demo.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add workspace root to sys.path for direct script execution
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from src.reranking import (
    build_reranking_report,
    explain_cost_latency_tradeoffs,
    rerank_candidates,
    retrieve_candidates,
    show,
)


def main() -> None:
    query = "What evidence is required for project submission?"
    final_k = 3

    print("=" * 60)
    print("DEMO: RAG Candidate Retrieval & Re-ranking")
    print("=" * 60)
    print(f"Query: '{query}'")
    print(f"Retrieving initial candidate pool size k = 10")
    print(f"Target final context size final_k = {final_k}\n")

    # 1. Stage 1: Retrieve candidate set larger than final k
    candidates = retrieve_candidates(query, k=10)

    print("--- Initial Order (Top final_k before re-ranking) ---")
    for rank, item in enumerate(candidates[:final_k], start=1):
        print(rank, item["score"], item["metadata"]["source"], item["text"][:100])
    print("\n" + "=" * 60 + "\n")

    # 2. Stage 2: Re-rank candidates by relevance
    final_context = rerank_candidates(query, candidates, final_k=final_k)

    # 3. Compare before and after
    show("before re-ranking", candidates[:final_k])
    print()
    show("after re-ranking", final_context)

    # 4. Explain cost & latency trade-offs
    print("\n" + "=" * 60)
    print("COST & LATENCY TRADE-OFF ANALYSIS")
    print("=" * 60)
    tradeoffs = explain_cost_latency_tradeoffs()
    print(json.dumps(tradeoffs, indent=2))

    # 5. Build and save markdown report
    report = build_reranking_report(
        query=query,
        candidates=candidates,
        final_context=final_context,
        final_k=final_k,
    )

    output_path = Path("outputs/reranking_demo.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Report saved successfully to {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

