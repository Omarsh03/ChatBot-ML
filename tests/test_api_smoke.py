from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from app.api import main as api_main


def test_api_health_ingest_chat_smoke(tmp_path: Path) -> None:
    original_index_dir = api_main.settings.index_dir
    original_transcripts_dir = api_main.settings.transcripts_dir
    original_top_k = api_main.settings.top_k
    original_min_score = api_main.settings.min_retrieval_score
    original_embedding_provider = api_main.settings.embedding_provider
    original_use_llm_grounded_answers = api_main.settings.use_llm_grounded_answers

    try:
        api_main.settings.index_dir = tmp_path
        api_main.settings.transcripts_dir = Path("data/transcripts")
        api_main.settings.top_k = 3
        api_main.settings.min_retrieval_score = 0.05
        api_main.settings.embedding_provider = "local_hash"
        api_main.settings.use_llm_grounded_answers = False

        client = TestClient(api_main.app)

        health_res = client.get("/health")
        assert health_res.status_code == 200
        assert health_res.json()["status"] == "ok"

        ingest_res = client.post("/ingest")
        assert ingest_res.status_code == 200
        ingest_payload = ingest_res.json()
        assert ingest_payload["documents"] >= 1
        assert ingest_payload["indexed"] >= 1

        chat_res = client.post(
            "/chat",
            json={"question": "What is supervised learning?"},
        )
        assert chat_res.status_code == 200
        chat_payload = chat_res.json()
        assert chat_payload["grounded"] is True
        assert len(chat_payload["citations"]) >= 1
        assert chat_payload["citations"][0]["lecture_title"]

        # Force insufficient-evidence path through a strict threshold.
        api_main.settings.min_retrieval_score = 1.1
        insufficient_res = client.post(
            "/chat",
            json={"question": "Any answer?"},
        )
        assert insufficient_res.status_code == 200
        insufficient_payload = insufficient_res.json()
        assert insufficient_payload["grounded"] is False
        assert insufficient_payload["citations"] == []
        assert insufficient_payload["answer"] == api_main.settings.insufficient_evidence_message
    finally:
        api_main.settings.index_dir = original_index_dir
        api_main.settings.transcripts_dir = original_transcripts_dir
        api_main.settings.top_k = original_top_k
        api_main.settings.min_retrieval_score = original_min_score
        api_main.settings.embedding_provider = original_embedding_provider
        api_main.settings.use_llm_grounded_answers = original_use_llm_grounded_answers


def test_ingest_returns_404_when_transcript_folder_missing(tmp_path: Path) -> None:
    original_transcripts_dir = api_main.settings.transcripts_dir
    original_index_dir = api_main.settings.index_dir
    try:
        api_main.settings.transcripts_dir = tmp_path / "missing_transcripts_root"
        api_main.settings.index_dir = tmp_path / "index"
        client = TestClient(api_main.app)

        response = client.post("/ingest")
        assert response.status_code == 404
        assert "Transcript directory not found" in response.json()["detail"]
    finally:
        api_main.settings.transcripts_dir = original_transcripts_dir
        api_main.settings.index_dir = original_index_dir


def test_transcribe_endpoint_calls_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run_transcribe_and_ingest(settings, course_id=None, run_ingestion=True, transcription_engine=None):
        _ = settings
        _ = transcription_engine
        assert course_id == "machine_learning"
        assert run_ingestion is True
        return {"media_files": 2, "transcribed": 2, "indexed": 10}

    monkeypatch.setattr(api_main, "run_transcribe_and_ingest", _fake_run_transcribe_and_ingest)
    client = TestClient(api_main.app)

    response = client.post("/transcribe", json={"course_id": "machine_learning", "run_ingestion": True})
    assert response.status_code == 200
    assert response.json() == {"media_files": 2, "transcribed": 2, "indexed": 10}


def test_chat_image_endpoint_uses_image_context(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_build_image_context_for_question(question, image_bytes, mime_type, settings):
        _ = image_bytes
        _ = mime_type
        _ = settings
        assert question == "What does this formula mean?"
        return "Extracted formula: P(A|B)=P(B|A)P(A)/P(B)"

    def _fake_run_retrieve_and_answer(
        question,
        settings,
        course_id=None,
        chat_history=None,
        retrieval_question=None,
        additional_context="",
        image_bytes=b"",
        image_mime_type="image/png",
    ):
        _ = settings
        _ = course_id
        _ = chat_history
        _ = image_mime_type
        assert question == "What does this formula mean?"
        assert retrieval_question is not None
        assert "Extracted formula" in retrieval_question
        assert "Extracted formula" in additional_context
        assert image_bytes == b"fake-image-bytes"
        return {"answer": "ok", "citations": [], "grounded": True}

    monkeypatch.setattr(api_main, "build_image_context_for_question", _fake_build_image_context_for_question)
    monkeypatch.setattr(api_main, "run_retrieve_and_answer", _fake_run_retrieve_and_answer)
    client = TestClient(api_main.app)

    response = client.post(
        "/chat_image",
        data={"question": "What does this formula mean?", "chat_history_json": "[]"},
        files={"image": ("formula.png", b"fake-image-bytes", "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "ok"


def test_chat_image_returns_clear_failure_when_vision_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_build_image_context_for_question(question, image_bytes, mime_type, settings):
        _ = question
        _ = image_bytes
        _ = mime_type
        _ = settings
        return ""

    def _fake_run_retrieve_and_answer(*args, **kwargs):
        raise AssertionError("retrieve_and_answer should not run for image-centric question without image context")

    monkeypatch.setattr(api_main, "build_image_context_for_question", _fake_build_image_context_for_question)
    monkeypatch.setattr(api_main, "run_retrieve_and_answer", _fake_run_retrieve_and_answer)
    client = TestClient(api_main.app)

    response = client.post(
        "/chat_image",
        data={"question": "תסביר לי את מה שיש בתמונה", "chat_history_json": "[]"},
        files={"image": ("formula.png", b"fake-image-bytes", "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["grounded"] is False
    assert payload["citations"] == []
    assert "לא הצלחתי לקרוא את התמונה" in payload["answer"]


def test_chat_endpoint_passes_image_context_for_followup(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run_retrieve_and_answer(
        question,
        settings,
        course_id=None,
        chat_history=None,
        additional_context="",
        image_bytes=b"",
        image_mime_type="image/png",
    ):
        _ = settings
        _ = course_id
        _ = chat_history
        _ = image_bytes
        _ = image_mime_type
        assert question == "תסביר בעברית"
        assert "Probability formulas" in additional_context
        return {
            "answer": "ok",
            "citations": [],
            "grounded": True,
            "image_context": additional_context,
        }

    monkeypatch.setattr(api_main, "run_retrieve_and_answer", _fake_run_retrieve_and_answer)
    client = TestClient(api_main.app)

    response = client.post(
        "/chat",
        json={
            "question": "תסביר בעברית",
            "chat_history": [{"role": "user", "content": "תסביר את מה שיש בתמונה"}],
            "image_context": "Summary of Probability formulas from uploaded image.",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "ok"
    assert "Probability formulas" in payload["image_context"]
