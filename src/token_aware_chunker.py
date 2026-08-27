from dataclasses import dataclass

import tiktoken


@dataclass(frozen=True)
class TokenChunk:
    index: int
    text: str
    token_count: int
    start_token: int
    end_token: int


class TokenAwareChunker:
    """Split text into model-token-sized chunks with controlled overlap."""

    def __init__(
        self,
        chunk_size: int = 120,
        overlap: int = 20,
        model: str = "gpt-4o-mini",
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.model = model
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def chunk(self, text: str) -> list[TokenChunk]:
        tokens = self.encoding.encode(text)
        chunks: list[TokenChunk] = []
        start = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(
                TokenChunk(
                    index=len(chunks),
                    text=self.encoding.decode(chunk_tokens),
                    token_count=len(chunk_tokens),
                    start_token=start,
                    end_token=end,
                )
            )
            if end == len(tokens):
                break
            start = end - self.overlap

        return chunks


def compare_boundary_context(text: str, chunk_size: int = 12, overlap: int = 0) -> list[TokenChunk]:
    """Return chunks for a compact boundary-context demonstration."""
    return TokenAwareChunker(chunk_size=chunk_size, overlap=overlap).chunk(text)