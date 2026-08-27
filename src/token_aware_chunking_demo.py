from pathlib import Path

from token_aware_chunker import TokenAwareChunker, compare_boundary_context


SAMPLE_TEXT = (
    "Before administering warfarin, verify the patient's current INR and review "
    "the medication list. If the INR is above the target range, hold the dose "
    "and contact the anticoagulation service for same-day guidance. Document "
    "the decision and reassess bleeding risk before restarting therapy."
)


def main() -> None:
    chunker = TokenAwareChunker(chunk_size=120, overlap=20)
    chunks = chunker.chunk(SAMPLE_TEXT)
    no_overlap = compare_boundary_context(SAMPLE_TEXT, chunk_size=12, overlap=0)
    with_overlap = compare_boundary_context(SAMPLE_TEXT, chunk_size=12, overlap=4)

    print("Token-aware chunking demo")
    print(f"Model encoding: {chunker.model} / cl100k_base-compatible")
    print(f"Settings: chunk_size={chunker.chunk_size} tokens, overlap={chunker.overlap} tokens")
    print(f"Total chunks: {len(chunks)}")
    for item in chunks:
        print(f"Chunk {item.index}: {item.token_count} tokens | {item.text}")

    print("\nBoundary comparison (12-token chunks)")
    print(f"Without overlap: {no_overlap[1].text!r}")
    print(f"With 4-token overlap: {with_overlap[1].text!r}")


if __name__ == "__main__":
    main()