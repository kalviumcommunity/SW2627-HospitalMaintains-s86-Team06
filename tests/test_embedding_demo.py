from src.embedding_demo import (
    SAMPLE_CORPUS,
    SAMPLE_QUERY,
    SAMPLE_TEXTS,
    build_report,
    cosine_similarity,
    embed_query,
    generate_embeddings,
    store_embeddings,
    rank_chunks,
    top_k_similarity_search,
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


def test_embed_query_and_top_k_similarity_search_include_scores_and_metadata():
    _, vectors = generate_embeddings(SAMPLE_TEXTS)
    records = store_embeddings(SAMPLE_CORPUS, vectors)

    provider, query_vector = embed_query(SAMPLE_QUERY)
    results = top_k_similarity_search(query_vector, records, k=2)

    assert provider == "offline semantic fixture"
    assert len(results) == 2
    assert all("score" in result and "text" in result and "metadata" in result for result in results)
    assert results[0]["metadata"]["source_document"] == "medication-guideline.pdf"
    assert results[0]["score"] >= results[1]["score"]


def test_top_k_results_change_when_k_changes():
    _, vectors = generate_embeddings(SAMPLE_TEXTS)
    records = store_embeddings(SAMPLE_CORPUS, vectors)
    _, query_vector = embed_query(SAMPLE_QUERY)

    top_two = top_k_similarity_search(query_vector, records, k=2)
    top_three = top_k_similarity_search(query_vector, records, k=3)

    assert len(top_two) == 2
    assert len(top_three) == 3
    assert top_two[0]["text"] == top_three[0]["text"]
    assert top_two[-1]["text"] != top_three[-1]["text"]