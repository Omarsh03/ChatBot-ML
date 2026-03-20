import json
import re

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.core.config import get_settings
from app.domain.models import ChatAnswer, ChatRequest, ChatTurn, IngestRequest, TranscribeRequest
from app.pipelines.ingest_and_index import run_ingest_and_index
from app.pipelines.transcribe_and_ingest import run_transcribe_and_ingest
from app.pipelines.retrieve_and_answer import run_retrieve_and_answer
from app.services.image_context_service import build_image_context_for_question

settings = get_settings()
app = FastAPI(title=settings.app_name)
_HE_PATTERN = re.compile(r"[\u0590-\u05FF]")
_IMAGE_CENTRIC_PATTERNS = [
    re.compile(r"תמונה"),
    re.compile(r"בתמונה"),
    re.compile(r"בצילום"),
    re.compile(r"הסבר(?:י)? לי (?:את )?(?:מה שיש|מה יש) בתמונה"),
    re.compile(r"\bimage\b", re.IGNORECASE),
    re.compile(r"\bthis image\b", re.IGNORECASE),
    re.compile(r"\bexplain (?:the|this) image\b", re.IGNORECASE),
    re.compile(r"\bwhat(?:'s| is) in (?:the|this) image\b", re.IGNORECASE),
]


def _parse_chat_history_json(payload: str) -> list[ChatTurn]:
    if not payload.strip():
        return []
    rows = json.loads(payload)
    return [ChatTurn.model_validate(row) for row in rows]


def _is_image_centric_question(question: str) -> bool:
    text = question.strip()
    return any(pattern.search(text) for pattern in _IMAGE_CENTRIC_PATTERNS)


def _image_read_failure_message(question: str) -> str:
    if _HE_PATTERN.search(question):
        return "לא הצלחתי לקרוא את התמונה בצורה אמינה. אפשר להעלות שוב תמונה חדה יותר או לשאול שאלה טקסטואלית מפורטת עליה."
    return (
        "I could not reliably read the uploaded image. "
        "Please re-upload a clearer image or ask a more specific text question about it."
    )


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
            additional_context=payload.image_context,
            settings=settings,
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc


@app.post("/chat_image", response_model=ChatAnswer)
async def chat_image(
    question: str = Form(...),
    course_id: str | None = Form(default=None),
    chat_history_json: str = Form(default="[]"),
    image: UploadFile = File(...),
) -> ChatAnswer:
    try:
        chat_history = _parse_chat_history_json(chat_history_json)
        image_bytes = await image.read()
        image_context = build_image_context_for_question(
            question=question,
            image_bytes=image_bytes,
            mime_type=image.content_type or "image/png",
            settings=settings,
        )
        if not image_context.strip() and _is_image_centric_question(question):
            return ChatAnswer(
                answer=_image_read_failure_message(question),
                citations=[],
                grounded=False,
            )
        retrieval_question = question
        if image_context:
            retrieval_question = f"{question}\n\nImage context:\n{image_context}"
        return run_retrieve_and_answer(
            question=question,
            course_id=course_id,
            chat_history=chat_history,
            retrieval_question=retrieval_question,
            additional_context=image_context,
            image_bytes=image_bytes,
            image_mime_type=image.content_type or "image/png",
            settings=settings,
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Chat image failed: {exc}") from exc
