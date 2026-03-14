from dataclasses import dataclass
import json
from pathlib import Path
import re

_MEDIA_EXTS = {".mp3", ".mp4", ".m4a", ".wav", ".webm", ".mkv"}
_LEC_ID_PATTERN = re.compile(r"^(lec[_-]?\d+)", re.IGNORECASE)


@dataclass
class LectureMediaItem:
    media_path: Path
    lecture_id: str
    lecture_title: str


def slugify(value: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", value.strip().lower(), flags=re.UNICODE)
    slug = re.sub(r"[\s-]+", "_", slug)
    return slug.strip("_")


def infer_lecture_from_filename(path: Path) -> tuple[str, str]:
    stem = path.stem
    match = _LEC_ID_PATTERN.match(stem)
    if match:
        lecture_id = match.group(1).replace("-", "_").lower()
        lecture_title = stem[match.end() :].strip(" _-")
        lecture_title = lecture_title.replace("_", " ").strip() or lecture_id
        return lecture_id, lecture_title
    lecture_id = slugify(stem)
    return lecture_id, stem.replace("_", " ").strip()


def load_manifest(manifest_path: Path) -> list[LectureMediaItem]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    items: list[LectureMediaItem] = []
    for entry in payload:
        media_path = Path(entry["media_path"])
        if not media_path.is_absolute():
            media_path = (manifest_path.parent / media_path).resolve()
        items.append(
            LectureMediaItem(
                media_path=media_path,
                lecture_id=entry["lecture_id"],
                lecture_title=entry["lecture_title"],
            )
        )
    return items


def discover_media_items(course_media_dir: Path, manifest_path: Path) -> list[LectureMediaItem]:
    if manifest_path.exists():
        items = load_manifest(manifest_path)
        return [item for item in items if item.media_path.exists()]

    items: list[LectureMediaItem] = []
    for path in sorted(course_media_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in _MEDIA_EXTS:
            continue
        lecture_id, lecture_title = infer_lecture_from_filename(path)
        items.append(
            LectureMediaItem(
                media_path=path,
                lecture_id=lecture_id,
                lecture_title=lecture_title,
            )
        )
    return items
