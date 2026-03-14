from pathlib import Path

from app.services.chunk_service import chunk_documents
from app.services.ingest_service import infer_lecture_id, load_course_transcripts


def test_infer_lecture_id() -> None:
    assert infer_lecture_id("lec_01_intro_to_ml.txt") == "lec_01"
    assert infer_lecture_id("lec-02-linear-regression.txt") == "lec_02"


def test_load_and_chunk_mock_transcripts() -> None:
    transcripts_root = Path("data/transcripts")
    metadata_root = Path("data/metadata")
    docs = load_course_transcripts(transcripts_root, "machine_learning", metadata_root)
    assert len(docs) >= 1

    chunks = chunk_documents(docs, chunk_size=120, chunk_overlap=20)
    assert len(chunks) >= len(docs)
    first = chunks[0]
    assert first.metadata.course_id == "machine_learning"
    assert first.metadata.lecture_title
    assert first.metadata.source_file.endswith(".txt")
    assert first.metadata.chunk_id.startswith(f"{first.metadata.lecture_id}_chunk_")
