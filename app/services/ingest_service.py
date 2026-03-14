from pathlib import Path
import json
import re

from app.domain.models import TranscriptDocument


_LECTURE_ID_PATTERN = re.compile(r"^(lec[_-]?\d+)", re.IGNORECASE)


def infer_lecture_id(source_file: str) -> str:
    stem = Path(source_file).stem
    match = _LECTURE_ID_PATTERN.match(stem)
    if match:
        return match.group(1).replace("-", "_").lower()
    return stem.lower().replace(" ", "_")


def _load_lecture_metadata_map(metadata_root: Path, course_id: str) -> dict[str, dict]:
    metadata_path = metadata_root / course_id / "lectures.json"
    if not metadata_path.exists():
        return {}
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_map: dict[str, dict] = {}
    for row in payload:
        source_file = row.get("source_file")
        if source_file:
            source_map[source_file] = row
    return source_map


def _infer_lecture_title(source_file: str, lecture_id: str) -> str:
    stem = Path(source_file).stem
    # Handle generated names like lec_01__intro_to_ml
    if "__" in stem:
        _, title = stem.split("__", 1)
        title = title.replace("_", " ").strip()
        if title:
            return title
    # Handle names like lec_01_intro_to_ml
    stripped = re.sub(rf"^{re.escape(lecture_id)}[_-]?", "", stem, flags=re.IGNORECASE).strip("_- ")
    return stripped.replace("_", " ").strip() or lecture_id


def load_course_transcripts(transcripts_root: Path, course_id: str, metadata_root: Path) -> list[TranscriptDocument]:
    course_dir = transcripts_root / course_id
    if not course_dir.exists():
        raise FileNotFoundError(f"Transcript directory not found: {course_dir}")
    lecture_meta_map = _load_lecture_metadata_map(metadata_root, course_id)

    documents: list[TranscriptDocument] = []
    for path in sorted(course_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        lecture_id = infer_lecture_id(path.name)
        source_meta = lecture_meta_map.get(path.name, {})
        lecture_title = source_meta.get("lecture_title") or _infer_lecture_title(path.name, lecture_id)
        documents.append(
            TranscriptDocument(
                course_id=course_id,
                lecture_id=lecture_id,
                lecture_title=lecture_title,
                source_file=path.name,
                text=text,
            )
        )
    return documents
