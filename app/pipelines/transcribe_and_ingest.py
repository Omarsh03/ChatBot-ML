import json
from pathlib import Path

from app.core.config import Settings
from app.pipelines.ingest_and_index import run_ingest_and_index
from app.transcription.engines.base import TranscriptionEngine
from app.transcription.manifest import LectureMediaItem, discover_media_items, slugify


def _resolve_course_paths(settings: Settings, course_id: str) -> tuple[Path, Path, Path, Path]:
    media_dir = settings.media_dir / course_id
    transcripts_dir = settings.transcripts_dir / course_id
    metadata_dir = settings.metadata_dir / course_id
    manifest_path = metadata_dir / settings.transcription_manifest_name

    transcripts_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return media_dir, transcripts_dir, metadata_dir, manifest_path


def _build_transcript_filename(item: LectureMediaItem) -> str:
    title_slug = slugify(item.lecture_title) or "untitled"
    return f"{item.lecture_id}__{title_slug}.txt"


def _write_lecture_metadata(metadata_path: Path, records: list[dict]) -> None:
    metadata_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_transcribe_and_ingest(
    settings: Settings,
    course_id: str | None = None,
    transcription_engine: TranscriptionEngine | None = None,
    run_ingestion: bool = True,
) -> dict[str, int]:
    selected_course_id = course_id or settings.default_course_id
    if transcription_engine is None:
        from app.transcription.engines.whisper_he_engine import WhisperHeEngine

        engine: TranscriptionEngine = WhisperHeEngine()
    else:
        engine = transcription_engine
    media_dir, transcripts_dir, metadata_dir, manifest_path = _resolve_course_paths(settings, selected_course_id)

    if not media_dir.exists():
        raise FileNotFoundError(f"Media directory not found: {media_dir}")

    items = discover_media_items(media_dir, manifest_path)
    if not items:
        return {"media_files": 0, "transcribed": 0, "indexed": 0}

    lecture_records: list[dict] = []
    transcribed_count = 0
    for item in items:
        transcript_text, runtime = engine.transcribe(item.media_path)
        if not transcript_text.strip():
            continue
        transcript_filename = _build_transcript_filename(item)
        transcript_path = transcripts_dir / transcript_filename
        transcript_path.write_text(transcript_text.strip(), encoding="utf-8")
        transcribed_count += 1
        lecture_records.append(
            {
                "lecture_id": item.lecture_id,
                "lecture_title": item.lecture_title,
                "source_file": transcript_filename,
                "media_file": item.media_path.name,
                "runtime": runtime,
            }
        )

    lectures_metadata_path = metadata_dir / settings.transcription_lectures_metadata_name
    _write_lecture_metadata(lectures_metadata_path, lecture_records)

    indexed_count = 0
    if run_ingestion and transcribed_count > 0:
        stats = run_ingest_and_index(settings=settings, course_id=selected_course_id)
        indexed_count = stats["indexed"]

    return {
        "media_files": len(items),
        "transcribed": transcribed_count,
        "indexed": indexed_count,
    }
