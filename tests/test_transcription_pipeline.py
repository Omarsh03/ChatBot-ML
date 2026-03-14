import json
from pathlib import Path

from app.core.config import Settings
from app.pipelines.transcribe_and_ingest import run_transcribe_and_ingest
from app.services.chunk_service import chunk_documents
from app.services.ingest_service import load_course_transcripts
from app.transcription.engines.base import TranscriptionEngine


class FakeTranscriptionEngine(TranscriptionEngine):
    def transcribe(self, media_path: Path) -> tuple[str, str]:
        return f"Transcript for {media_path.stem}", "0.001 seconds"


def test_transcribe_pipeline_writes_transcripts_and_metadata(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    transcripts_dir = tmp_path / "transcripts"
    metadata_dir = tmp_path / "metadata"
    index_dir = tmp_path / "indexes"
    course_id = "machine_learning"

    course_media_dir = media_dir / course_id
    course_meta_dir = metadata_dir / course_id
    course_media_dir.mkdir(parents=True, exist_ok=True)
    course_meta_dir.mkdir(parents=True, exist_ok=True)

    media1 = course_media_dir / "lecture1.mp3"
    media2 = course_media_dir / "lecture2.mp4"
    media1.write_text("fake-audio-1", encoding="utf-8")
    media2.write_text("fake-audio-2", encoding="utf-8")

    manifest = [
        {"media_path": str(media1), "lecture_id": "lec_01", "lecture_title": "Intro"},
        {"media_path": str(media2), "lecture_id": "lec_02", "lecture_title": "Regression"},
    ]
    (course_meta_dir / "lectures_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    settings = Settings(
        DEFAULT_COURSE_ID=course_id,
        MEDIA_DIR=media_dir,
        TRANSCRIPTS_DIR=transcripts_dir,
        METADATA_DIR=metadata_dir,
        INDEX_DIR=index_dir,
        TRANSCRIPTION_MANIFEST_NAME="lectures_manifest.json",
        TRANSCRIPTION_LECTURES_METADATA_NAME="lectures.json",
    )

    stats = run_transcribe_and_ingest(
        settings=settings,
        transcription_engine=FakeTranscriptionEngine(),
        run_ingestion=False,
    )
    assert stats["media_files"] == 2
    assert stats["transcribed"] == 2
    assert stats["indexed"] == 0

    generated = sorted((transcripts_dir / course_id).glob("*.txt"))
    assert len(generated) == 2
    metadata_path = metadata_dir / course_id / "lectures.json"
    assert metadata_path.exists()


def test_ingest_uses_generated_lecture_title_metadata(tmp_path: Path) -> None:
    course_id = "machine_learning"
    transcripts_dir = tmp_path / "transcripts" / course_id
    metadata_dir = tmp_path / "metadata" / course_id
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    transcript_name = "lec_10__advanced_topics.txt"
    (transcripts_dir / transcript_name).write_text("Some lecture content", encoding="utf-8")
    (metadata_dir / "lectures.json").write_text(
        json.dumps(
            [
                {
                    "lecture_id": "lec_10",
                    "lecture_title": "Advanced Topics",
                    "source_file": transcript_name,
                }
            ]
        ),
        encoding="utf-8",
    )

    docs = load_course_transcripts(
        transcripts_root=tmp_path / "transcripts",
        course_id=course_id,
        metadata_root=tmp_path / "metadata",
    )
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)

    assert docs[0].lecture_title == "Advanced Topics"
    assert chunks[0].metadata.lecture_title == "Advanced Topics"
