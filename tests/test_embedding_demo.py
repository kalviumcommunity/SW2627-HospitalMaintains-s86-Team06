from src.embedding_demo import (
    SAMPLE_CORPUS,
    SAMPLE_QUERY,
    SAMPLE_TEXTS,
    build_report,
    cosine_similarity,
    generate_embeddings,
    store_embeddings,
    rank_chunks,
)


def test_embeddings_have_one_dimension_and_similar_pair_scores_higher():
    provider, vectors = generate_embeddings(SAMPLE_TEXTS)

    assert provider == "offline semantic fixture"
    assert len({len(vector) for vector in vectors}) == 1
    assert cosine_similarity(vectors[0], vectors[1]) > cosine_similarity(vectors[0], vectors[2])


def test_api_embeddings_are_requested_and_stored_with_metadata():
    class FakeEmbeddings:
        def create(self, *, input, model):
            assert input == SAMPLE_TEXTS
            assert model == "configured-embedding-model"
            return type("Response", (), {
                "data": [
                    type("Item", (), {"index": index, "embedding": [float(index), 1.0]})
                    for index in range(len(input))
                ]
            })()

    fake_client = type("Client", (), {"embeddings": FakeEmbeddings()})()
    provider, vectors = generate_embeddings(
        SAMPLE_TEXTS, client=fake_client, model="configured-embedding-model"
    )
    records = store_embeddings(SAMPLE_CORPUS, vectors)

    assert provider == "OpenAI configured-embedding-model"
    assert len(records) == 3
    assert records[0]["text"] == SAMPLE_CORPUS[0].text
    assert records[0]["metadata"]["source_document"] == "medication-guideline.pdf"
    assert len(records[0]["embedding"]) == 2


def test_report_contains_vectors_dimension_and_explanation():
    provider, vectors = generate_embeddings(SAMPLE_TEXTS)
    report = build_report(provider, SAMPLE_CORPUS, vectors)

    assert "Chunks embedded: `3`" in report
    assert "**Vector length:** `8`" in report
    assert "Similar pair scores higher: `True`" in report
    assert "source_document" in report
    assert "not random IDs" in report


def test_query_ranks_chunks_with_scores_and_metadata():
    _, vectors = generate_embeddings(SAMPLE_TEXTS)
    records = store_embeddings(SAMPLE_CORPUS, vectors)
    _, query_vectors = generate_embeddings([SAMPLE_QUERY])

    rankings = rank_chunks(query_vectors[0], records)

    assert len(rankings) == 3
    assert rankings[0]["metadata"]["section"] == "Dosage"
    assert rankings[-1]["metadata"]["section"] == "Account access"
    assert rankings[0]["score"] > rankings[-1]["score"]


def test_known_relevance_cases_rank_related_chunks_above_unrelated_ones():
    from src.retrieval_quality_check import KNOWN_CASES, evaluate_case

    assert len(KNOWN_CASES) >= 3

    outcomes = [evaluate_case(case) for case in KNOWN_CASES]

    assert all(result["related_above_unrelated"] for result in outcomes)
    assert all(result["expected_top_ranked"] for result in outcomes)
    assert all(result["top_result"]["source_document"] == case["expected_top"]["source_document"] for result, case in zip(outcomes, KNOWN_CASES))