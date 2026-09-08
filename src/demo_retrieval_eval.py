"""Demonstration script for Quantitative Retrieval Evaluation (Recall & Precision).

Executes:
1. Labelled query dataset evaluation (recall@5, precision@5).
2. Failure inspection and root cause diagnosis for queries with recall < 1.0.
3. Generating a Markdown report saved to outputs/retrieval_eval_report.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add workspace root to sys.path for direct script execution
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from src.retrieval_eval import (
    LABELLED_QUERIES,
    build_evaluation_report,
    evaluate_dataset,
    inspect_failures,
)


def main() -> None:
    k = 5

    print("=" * 60)
    print("DEMO: Quantitative Retrieval Quality Evaluation")
    print("=" * 60)
    print(f"Evaluating {len(LABELLED_QUERIES)} labelled queries at k = {k}\n")

    # 1. Measure recall@k and precision@k across dataset
    eval_summary = evaluate_dataset(LABELLED_QUERIES, k=k)

    print("queries:", eval_summary["total_queries"])
    print(f"recall@{k}:", round(eval_summary["avg_recall"], 3))
    print(f"precision@{k}:", round(eval_summary["avg_precision"], 3))
    print(f"failures:", len(eval_summary["failures"]))

    # 2. Inspect failures if any exist
    failures = eval_summary["failures"]
    if failures:
        print("\n" + "=" * 60)
        print("FAILURE INSPECTION (Recall < 1.0)")
        print("=" * 60)
        for failure in failures:
            print("failed query:", failure["query"])
            print("expected:", failure["relevant_chunk_ids"])
            print("retrieved:", failure["retrieved_ids"])
            print("recall:", failure["recall"])
            print("precision:", failure["precision"])
            print("-" * 40)

        diagnostics = inspect_failures(failures)
        print("\n--- Root Cause Diagnostics ---")
        for diag in diagnostics:
            print(f"Query: {diag['query']}")
            print(f"Missing: {diag['missing_ids']}")
            print(f"Diagnosis: {diag['root_cause_diagnosis']}\n")
    else:
        print("\nAll queries achieved 100% recall@k! No retrieval failures detected.")

    # 3. Generate & save evaluation report
    report = build_evaluation_report(eval_summary)
    output_path = Path("outputs/retrieval_eval_report.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Report saved successfully to {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
