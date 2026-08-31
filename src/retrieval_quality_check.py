"""Known-relevance checks for the offline embedding demo.

This script intentionally uses a tiny set of deterministic query vectors and
source chunks so that the retrieval quality can be checked without a live model.
The focus is to confirm that semantically related content ranks above clearly
unrelated content, while also recording any borderline or surprising cases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.embedding_demo import SAMPLE_CORPUS, OFFLINE_QUERY_VECTOR, OFFLINE_VECTORS, rank_chunks, store_embeddings

KNOWN_CASES = [
    {
        "name": "medication-guidance",
        "query": "What medication instructions should the patient follow?",
        "query_vector": OFFLINE_QUERY_VECTOR,
        "expected_top": {
            "text": "The patient should take the prescribed medication with water.",
            "source_document": "medication-guideline.pdf",
        },
        "unrelated_text": "The help desk can reset an employee password.",
        "note": "The dosage guidance should outrank the unrelated IT account-access content.",
    },
    {
        "name": "medication-paraphrase",
        "query": "How do patients use their recommended medicine with water?",
        "query_vector": [0.89, 0.10, 0.03, 0.11, 0.05, 0.02, 0.01, 0.03],
        "expected_top": {
            "text": "Patients need to use their recommended medicine with water.",
            "source_document": "medication-guideline.pdf",
        },
        "unrelated_text": "The help desk can reset an employee password.",
        "note": "This is a borderline paraphrase case: both medication chunks are very similar, so the top result should be close but still preferred.",
    },
    {
        "name": "account-access",
        "query": "How can a staff member reset a password?",
        "query_vector": [0.04, 0.02, 0.91, 0.03, 0.08, 0.02, 0.01, 0.05],
        "expected_top": {
            "text": "The help desk can reset an employee password.",
            "source_document": "it-support-handbook.pdf",
        },
        "unrelated_text": "The patient should take the prescribed medication with water.",
        "note": "An IT support query should prefer the account-access chunk over clinical guidance.",
    },
]


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    """Score a single known relevance case and return a structured result."""
    records = store_embeddings(SAMPLE_CORPUS, OFFLINE_VECTORS)
    rankings = rank_chunks(case["query_vector"], records)
    ranked_by_text = {result["text"]: result for result in rankings}

    expected_top_text = case["expected_top"]["text"]
    if expected_top_text not in ranked_by_text:
        raise ValueError(f"Expected chunk not found in ranking: {expected_top_text}")
    if case["unrelated_text"] not in ranked_by_text:
        raise ValueError(f"Unrelated chunk not found in ranking: {case['unrelated_text']}")

    top_result = rankings[0]
    expected_score = ranked_by_text[expected_top_text]["score"]
    unrelated_score = ranked_by_text[case["unrelated_text"]]["score"]
    second_result = rankings[1] if len(rankings) > 1 else top_result
    score_gap = top_result["score"] - second_result["score"]

    result = {
        "name": case["name"],
        "query": case["query"],
        "expected_top": case["expected_top"],
        "top_result": {
            "text": top_result["text"],
            "score": round(float(top_result["score"]), 6),
            "source_document": top_result["metadata"]["source_document"],
            "section": top_result["metadata"]["section"],
        },
        "related_above_unrelated": expected_score > unrelated_score,
        "expected_top_ranked": top_result["text"] == expected_top_text,
        "score_gap": round(float(score_gap), 6),
        "relevant_score": round(float(expected_score), 6),
        "unrelated_score": round(float(unrelated_score), 6),
        "note": case["note"],
    }
    return result


def build_sanity_report() -> str:
    """Generate a concise human-readable sanity report for all known relevance tests."""
    outcomes = [evaluate_case(case) for case in KNOWN_CASES]
    passed = sum(1 for outcome in outcomes if outcome["related_above_unrelated"] and outcome["expected_top_ranked"])
    failed = len(outcomes) - passed
    borderline_cases = [outcome for outcome in outcomes if outcome["score_gap"] < 0.02]

    lines = [
        "# Retrieval Quality Sanity Report",
        "",
        f"Test count: {len(outcomes)}",
        f"Passes: {passed}",
        f"Failures: {failed}",
        "",
        "## Top-ranked sources",
        "",
    ]

    for outcome in outcomes:
        lines.append(
            f"- {outcome['name']}: {outcome['top_result']['text']} "
            f"(score={outcome['top_result']['score']:.6f}, source={outcome['top_result']['source_document']})"
        )

    lines.extend(["", "## Result details", ""])
    for outcome in outcomes:
        lines.append(
            f"- {outcome['name']}: related_above_unrelated={outcome['related_above_unrelated']}, "
            f"expected_top_ranked={outcome['expected_top_ranked']}, "
            f"relevant_score={outcome['relevant_score']:.6f}, unrelated_score={outcome['unrelated_score']:.6f}, "
            f"score_gap={outcome['score_gap']:.6f}"
        )
        lines.append(f"  Note: {outcome['note']}")

    lines.extend(["", "## Borderline or surprising case", ""])
    if borderline_cases:
        for outcome in borderline_cases:
            lines.append(
                f"- {outcome['name']}: the top result is only {outcome['score_gap']:.6f} above the next rank, "
                "which indicates a borderline semantic match. This revealed a near-duplicate medication pair in the corpus, "
                "so the metric is still behaving sensibly but the corpus contains close paraphrases that can produce narrow score gaps."
            )
    else:
        lines.append("- No borderline cases were observed in this deterministic offline fixture.")

    return "\n".join(lines) + "\n"


def main() -> None:
    report = build_sanity_report()
    output_path = Path("outputs/retrieval_quality_report.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
