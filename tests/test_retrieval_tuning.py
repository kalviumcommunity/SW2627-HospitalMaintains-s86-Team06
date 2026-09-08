from src.retrieval_tuning import QUERIES, run_experiment


def test_tuning_queries_have_expected_sources():
    assert len(QUERIES) == 5
    assert all(query.expected_sources for query in QUERIES)


def test_adaptive_filter_is_perfect_and_beats_baseline():
    rows = run_experiment()
    averages = {row["setting"]: row for row in rows if row["query"] == "AVERAGE"}

    adaptive = averages["adaptive filter: k=2, query-specific section, threshold=0.80"]
    baseline = averages["baseline: k=3, no filter"]

    assert adaptive["top1_hit"] == 1.0
    assert adaptive["topk_hit"] == 1.0
    assert adaptive["reciprocal_rank"] == 1.0
    assert adaptive["top1_hit"] >= baseline["top1_hit"]


def test_wrong_fixed_filter_exposes_filter_risk():
    rows = run_experiment()
    account_filter_rows = [
        row
        for row in rows
        if row["setting"] == "filtered: k=2, section filter, threshold=0.80"
        and row["query"] != "AVERAGE"
    ]

    assert account_filter_rows[0]["topk_hit"] == 1.0
    assert any(row["topk_hit"] == 0.0 for row in account_filter_rows[1:])