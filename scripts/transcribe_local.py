"""
Manual local transcription entry-point.

Flow:
- read local lecture media files for configured course
- transcribe with WhisperHe engine
- write one transcript text file per lecture
- write lectures metadata
- run ingestion/indexing
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.pipelines.transcribe_and_ingest import run_transcribe_and_ingest


def _progress_bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[no items]"
    ratio = min(max(done / total, 0.0), 1.0)
    filled = int(round(width * ratio))
    percent = int(ratio * 100)
    return f"[{'#' * filled}{'-' * (width - filled)}] {done}/{total} ({percent}%)"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe local media files and optionally ingest/index.")
    parser.add_argument(
        "--course-id",
        default=None,
        help="Course ID to process (defaults to DEFAULT_COURSE_ID from .env).",
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Only transcribe media files without ingest/index.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = get_settings()

    def _on_progress(done: int, total: int, item, success: bool) -> None:
        state = "ok" if success else "empty"
        print(f"{_progress_bar(done, total)}  {state:5}  {item.media_path.name}", flush=True)

    stats = run_transcribe_and_ingest(
        settings,
        course_id=args.course_id,
        run_ingestion=not args.skip_ingestion,
        progress_callback=_on_progress,
    )
    print(f"Media files discovered: {stats['media_files']}")
    print(f"Transcripts generated: {stats['transcribed']}")
    print(f"Indexed chunks: {stats['indexed']}")


if __name__ == "__main__":
    main()
