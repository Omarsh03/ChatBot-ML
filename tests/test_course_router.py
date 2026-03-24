from pathlib import Path

from app.core.config import Settings
from app.services import course_router


def test_question_hint_routes_to_probability(tmp_path: Path) -> None:
    prob_dir = tmp_path / "probability"
    ml_dir = tmp_path / "machine_learning"
    prob_dir.mkdir(parents=True, exist_ok=True)
    ml_dir.mkdir(parents=True, exist_ok=True)
    (prob_dir / "index.faiss").write_text("x", encoding="utf-8")
    (prob_dir / "chunks.jsonl").write_text("x", encoding="utf-8")
    (ml_dir / "index.faiss").write_text("x", encoding="utf-8")
    (ml_dir / "chunks.jsonl").write_text("x", encoding="utf-8")

    settings = Settings(INDEX_DIR=tmp_path, EMBEDDING_PROVIDER="local_hash")
    selected = course_router.choose_course_id(
        question="תסביר את חוק בייס עם דוגמה",
        settings=settings,
        explicit_course_id=None,
    )
    assert selected == "probability"


def test_explicit_course_id_bypasses_auto_routing(tmp_path: Path) -> None:
    settings = Settings(INDEX_DIR=tmp_path, EMBEDDING_PROVIDER="local_hash")
    selected = course_router.choose_course_id(
        question="Whatever question",
        settings=settings,
        explicit_course_id="machine_learning",
    )
    assert selected == "machine_learning"


def test_question_hint_routes_to_machine_learning_for_supervised_and_linear_terms(tmp_path: Path) -> None:
    prob_dir = tmp_path / "probability"
    ml_dir = tmp_path / "machine_learning"
    prob_dir.mkdir(parents=True, exist_ok=True)
    ml_dir.mkdir(parents=True, exist_ok=True)
    (prob_dir / "index.faiss").write_text("x", encoding="utf-8")
    (prob_dir / "chunks.jsonl").write_text("x", encoding="utf-8")
    (ml_dir / "index.faiss").write_text("x", encoding="utf-8")
    (ml_dir / "chunks.jsonl").write_text("x", encoding="utf-8")

    settings = Settings(INDEX_DIR=tmp_path, EMBEDDING_PROVIDER="local_hash")
    selected_supervised = course_router.choose_course_id(
        question="מה זה למידה מונחית?",
        settings=settings,
        explicit_course_id=None,
    )
    selected_linear = course_router.choose_course_id(
        question="מה המשמעות של linear separability?",
        settings=settings,
        explicit_course_id=None,
    )
    assert selected_supervised == "machine_learning"
    assert selected_linear == "machine_learning"
