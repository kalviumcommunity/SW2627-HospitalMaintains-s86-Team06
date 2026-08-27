import json

from src.ingestion_pipeline import run_ingestion


def test_ingestion_reports_successes_failures_and_samples(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    data_dir.mkdir()
    (data_dir / "policy.txt").write_text(
        "Overview of policy.\n\nFollow the clinical procedure.", encoding="utf-8"
    )
    (data_dir / "notes.zip").write_bytes(b"not a supported document")

    summary = run_ingestion(data_dir, output_dir)

    assert summary["total_source_documents"] == 2
    assert summary["successfully_ingested_documents"] == 1
    assert summary["total_chunks_created"] == 2
    assert summary["failure_count"] == 1
    assert summary["completeness_check"]["passed"] is True
    assert summary["completeness_check"]["source_documents"] == 2
    assert summary["completeness_check"]["ingested_plus_failures"] == 2

    samples = json.loads((output_dir / "sample_chunks.json").read_text(encoding="utf-8"))
    assert samples[0]["source"] == "policy.txt"
    assert samples[0]["chunk_index"] == 1
    assert samples[0]["strategy"] == "paragraph"
    assert samples[0]["text"] == "Overview of policy."
    assert samples[0]["char_count"] == len(samples[0]["text"])