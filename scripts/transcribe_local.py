"""
Manual local transcription entry-point.

Flow:
- read local lecture media files for configured course
- transcribe with WhisperHe engine
- write one transcript text file per lecture
- write lectures metadata
- run ingestion/indexing
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.pipelines.transcribe_and_ingest import run_transcribe_and_ingest


def main() -> None:
    settings = get_settings()
    stats = run_transcribe_and_ingest(settings)
    print(f"Media files discovered: {stats['media_files']}")
    print(f"Transcripts generated: {stats['transcribed']}")
    print(f"Indexed chunks: {stats['indexed']}")


if __name__ == "__main__":
    main()
