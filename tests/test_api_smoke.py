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
            json={"question": "What is supervised learning?", "course_id": "machine_learning"},
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
            json={"question": "Any answer?", "course_id": "machine_learning"},
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
