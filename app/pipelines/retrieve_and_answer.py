from app.core.config import Settings
from app.domain.models import ChatAnswer, ChatTurn, DocumentChunk
from app.services.answer_service import generate_grounded_answer
from app.services.conversation_context import build_retrieval_question, is_contextual_followup
from app.services.course_router import choose_course_id
from app.services.retrieve_service import retrieve_chunks


def _merge_retrieval_hits(
    primary_hits: list[tuple[DocumentChunk, float]],
    secondary_hits: list[tuple[DocumentChunk, float]],
    secondary_weight: float = 0.92,
) -> list[tuple[DocumentChunk, float]]:
    merged: dict[str, tuple[DocumentChunk, float]] = {}
    for chunk, score in primary_hits:
        merged[chunk.metadata.chunk_id] = (chunk, score)
    for chunk, score in secondary_hits:
        boosted = score * secondary_weight
        key = chunk.metadata.chunk_id
        if key in merged:
            existing_chunk, existing_score = merged[key]
            merged[key] = (existing_chunk, max(existing_score, boosted) + 0.01)
        else:
            merged[key] = (chunk, boosted)
    return sorted(merged.values(), key=lambda pair: pair[1], reverse=True)


def run_retrieve_and_answer(
    question: str,
    settings: Settings,
    course_id: str | None = None,
    chat_history: list[ChatTurn] | None = None,
    retrieval_question: str | None = None,
    additional_context: str = "",
    image_bytes: bytes = b"",
    image_mime_type: str = "image/png",
) -> ChatAnswer:
    history = chat_history or []
    followup = is_contextual_followup(question)
    effective_image_context = additional_context.strip() if (retrieval_question is not None or followup) else ""
    retrieval_text = retrieval_question or build_retrieval_question(question=question, chat_history=history)
    if effective_image_context and "Recent image context:" not in retrieval_text:
        retrieval_text = f"{retrieval_text}\n\nRecent image context:\n{effective_image_context}"
    selected_course_id = choose_course_id(
        question=retrieval_text,
        settings=settings,
        explicit_course_id=course_id,
    )
    hits = retrieve_chunks(question=retrieval_text, settings=settings, course_id=selected_course_id)
    if effective_image_context:
        image_query = f"{question}\n\nImage details:\n{effective_image_context[:1600]}"
        image_hits = retrieve_chunks(question=image_query, settings=settings, course_id=selected_course_id)
        hits = _merge_retrieval_hits(hits, image_hits)
        hits = hits[: settings.top_k]
    result = generate_grounded_answer(
        question=question,
        hits=hits,
        settings=settings,
        chat_history=history,
        additional_context=effective_image_context,
        image_bytes=image_bytes,
        image_mime_type=image_mime_type,
    )
    result.image_context = effective_image_context
    return result
