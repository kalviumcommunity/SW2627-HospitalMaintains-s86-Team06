import pytest

from src.token_aware_chunker import TokenAwareChunker


def test_chunks_are_bounded_by_token_count():
    chunker = TokenAwareChunker(chunk_size=8, overlap=2)

    chunks = chunker.chunk("Clinical protocols should be reviewed before emergency use.")

    assert chunks
    assert all(chunk.token_count <= 8 for chunk in chunks)
    assert all(chunk.token_count == len(chunker.encoding.encode(chunk.text)) for chunk in chunks)


def test_adjacent_chunks_repeat_configured_overlap_tokens():
    chunker = TokenAwareChunker(chunk_size=8, overlap=2)

    chunks = chunker.chunk("one two three four five six seven eight nine ten eleven twelve")

    first_tokens = chunker.encoding.encode(chunks[0].text)
    second_tokens = chunker.encoding.encode(chunks[1].text)
    assert second_tokens[:2] == first_tokens[-2:]


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        TokenAwareChunker(chunk_size=10, overlap=10)