from pathlib import Path

from app.core.config import Settings
from app.pipelines.ingest_and_index import run_ingest_and_index
from app.pipelines.retrieve_and_answer import run_retrieve_and_answer


def test_retrieve_and_answer_smoke(tmp_path: Path) -> None:
    settings = Settings(
        DEFAULT_COURSE_ID="machine_learning",
        TRANSCRIPTS_DIR=Path("data/transcripts"),
        INDEX_DIR=tmp_path,
        USE_LLM_GROUNDED_ANSWERS=False,
        EMBEDDING_PROVIDER="local_hash",
        CHUNK_SIZE=120,
        CHUNK_OVERLAP=20,
        EMBEDDING_DIM=64,
        TOP_K=3,
        MIN_RETRIEVAL_SCORE=0.05,
        MIN_EVIDENCE_HITS=1,
        MIN_LEXICAL_OVERLAP=0.1,
        INSUFFICIENT_EVIDENCE_MESSAGE="INSUFFICIENT_TRANSCRIPT_EVIDENCE",
    )

    run_ingest_and_index(settings)

    grounded = run_retrieve_and_answer("What is supervised learning?", settings=settings)
    assert grounded.grounded is True
    assert len(grounded.citations) >= 1
    assert grounded.citations[0].source_file.endswith(".txt")

    insufficient = run_retrieve_and_answer("", settings=settings)
    assert insufficient.grounded is False
    assert insufficient.answer == "INSUFFICIENT_TRANSCRIPT_EVIDENCE"
    assert insufficient.citations == []

    unrelated = run_retrieve_and_answer("quantum entanglement in hadron collider", settings=settings)
    assert unrelated.grounded is False
    assert unrelated.answer == "INSUFFICIENT_TRANSCRIPT_EVIDENCE"
