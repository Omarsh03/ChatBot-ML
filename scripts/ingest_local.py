"""
Manual local transcript ingestion entry-point.

This script performs ingestion/indexing on existing transcript files:
- load local transcript files
- split into deterministic chunks
- embed and index chunks in FAISS
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.pipelines.ingest_and_index import run_ingest_and_index


def main() -> None:
    settings = get_settings()
    stats = run_ingest_and_index(settings)
    print(f"Loaded transcripts: {stats['documents']}")
    print(f"Generated chunks: {stats['chunks']}")
    print(f"Indexed chunks: {stats['indexed']}")
    print(f"Index location: {settings.index_dir / settings.default_course_id}")


if __name__ == "__main__":
    main()
