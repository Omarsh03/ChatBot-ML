from fastapi import FastAPI, HTTPException

from app.core.config import get_settings
from app.domain.models import ChatAnswer, ChatRequest, IngestRequest, TranscribeRequest
from app.pipelines.ingest_and_index import run_ingest_and_index
from app.pipelines.transcribe_and_ingest import run_transcribe_and_ingest
from app.pipelines.retrieve_and_answer import run_retrieve_and_answer

settings = get_settings()
app = FastAPI(title=settings.app_name)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "env": settings.app_env}


@app.post("/ingest")
def ingest(payload: IngestRequest = IngestRequest()) -> dict[str, int]:
    try:
        return run_ingest_and_index(settings, course_id=payload.course_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


@app.post("/transcribe")
def transcribe(payload: TranscribeRequest) -> dict[str, int]:
    try:
        return run_transcribe_and_ingest(
            settings=settings,
            course_id=payload.course_id,
            run_ingestion=payload.run_ingestion,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc


@app.post("/chat", response_model=ChatAnswer)
def chat(payload: ChatRequest) -> ChatAnswer:
    try:
        return run_retrieve_and_answer(
            question=payload.question,
            course_id=payload.course_id,
            chat_history=payload.chat_history,
            settings=settings,
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc
