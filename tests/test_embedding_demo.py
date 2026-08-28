from src.embedding_demo import (
    SAMPLE_TEXTS,
    build_report,
    cosine_similarity,
    generate_embeddings,
)


def test_embeddings_have_one_dimension_and_similar_pair_scores_higher():
    provider, vectors = generate_embeddings(SAMPLE_TEXTS)

    assert provider == "offline semantic fixture"
    assert len({len(vector) for vector in vectors}) == 1
    assert cosine_similarity(vectors[0], vectors[1]) > cosine_similarity(vectors[0], vectors[2])


def test_report_contains_vectors_dimension_and_explanation():
    provider, vectors = generate_embeddings(SAMPLE_TEXTS)
    report = build_report(provider, SAMPLE_TEXTS, vectors)

    assert "Vector dimension:" in report
    assert "Similar pair scores higher: `True`" in report
    assert "not random IDs" in report