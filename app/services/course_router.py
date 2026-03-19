from pathlib import Path
import re

from app.core.config import Settings
from app.providers.vectorstore.faiss_store import FaissVectorStore
from app.services.embedding_factory import build_embedding_provider

_QUESTION_HINTS: dict[str, tuple[re.Pattern[str], ...]] = {
    "machine_learning": (
        re.compile(r"\bmachine learning\b", re.IGNORECASE),
        re.compile(r"\bml\b", re.IGNORECASE),
        re.compile(r"למידת\s+מכונה"),
    ),
    "probability": (
        re.compile(r"\bprobability\b", re.IGNORECASE),
        re.compile(r"\bbayes(?:'|’)?\b", re.IGNORECASE),
        re.compile(r"ביי?ס"),
        re.compile(r"הסתברות"),
    ),
}


def _discover_indexed_courses(index_dir: Path) -> list[str]:
    if not index_dir.exists():
        return []
    course_ids: list[str] = []
    for entry in sorted(index_dir.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "index.faiss").exists() and (entry / "chunks.jsonl").exists():
            course_ids.append(entry.name)
    return course_ids


def _course_hint_from_question(question: str, available_courses: list[str]) -> str | None:
    text = question.strip()
    if not text:
        return None
    available = set(available_courses)
    for course_id, patterns in _QUESTION_HINTS.items():
        if course_id not in available:
            continue
        if any(pattern.search(text) for pattern in patterns):
            return course_id
    return None


def choose_course_id(
    question: str,
    settings: Settings,
    explicit_course_id: str | None = None,
) -> str:
    if explicit_course_id and explicit_course_id.strip() and explicit_course_id.strip().lower() != "auto":
        return explicit_course_id.strip()

    available_courses = _discover_indexed_courses(settings.index_dir)
    if not available_courses:
        return settings.default_course_id

    hinted = _course_hint_from_question(question, available_courses)
    if hinted:
        return hinted

    embedding_provider = build_embedding_provider(settings)
    query_vector = embedding_provider.embed_query(question)

    best_course = settings.default_course_id
    best_score = float("-inf")

    for course_id in available_courses:
        store = FaissVectorStore(settings.index_dir, course_id)
        hits = store.search(query_vector, top_k=1)
        if not hits:
            continue
        _, score = hits[0]
        if score > best_score:
            best_score = score
            best_course = course_id

    return best_course
