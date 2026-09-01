"""Sample chunks generator for testing and demonstration."""

from src.chunk_metadata import Chunk, ChunkMetadata, ChunkStore


def build_sample_chunks() -> ChunkStore:
    """Build a standard ChunkStore containing 3 sample chunks."""
    store = ChunkStore()
    store.add_chunk(
        text="The patient should take the prescribed medication with water.",
        source_id="medication-guideline.pdf",
        section="Dosage",
        page=1,
        position=0,
    )
    store.add_chunk(
        text="Patients need to use their recommended medicine with water.",
        source_id="medication-guideline.pdf",
        section="Administration",
        page=2,
        position=1,
    )
    store.add_chunk(
        text="The help desk can reset an employee password.",
        source_id="it-support-handbook.pdf",
        section="Account access",
        page=5,
        position=2,
    )
    return store
