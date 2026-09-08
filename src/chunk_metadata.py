"""Chunk metadata and storage dataclasses."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ChunkMetadata:
    """Metadata associated with a text chunk."""
    source_id: str
    section: Optional[str] = None
    page: Optional[int] = None
    position: Optional[int] = None


@dataclass
class Chunk:
    """Text chunk with metadata."""
    chunk_id: str
    text: str
    metadata: ChunkMetadata


class ChunkStore:
    """Store for managing text chunks."""
    def __init__(self) -> None:
        self._chunks: Dict[str, Chunk] = {}

    def add_chunk(
        self,
        text: str,
        source_id: str,
        section: Optional[str] = None,
        page: Optional[int] = None,
        position: Optional[int] = None,
    ) -> Chunk:
        chunk_id = f"{source_id}:{len(self._chunks)}"
        metadata = ChunkMetadata(source_id=source_id, section=section, page=page, position=position)
        chunk = Chunk(chunk_id=chunk_id, text=text, metadata=metadata)
        self._chunks[chunk_id] = chunk
        return chunk

    def list_chunks(self) -> List[Chunk]:
        return list(self._chunks.values())

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self._chunks.get(chunk_id)
