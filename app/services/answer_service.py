import logging

from app.core.config import Settings
from app.domain.models import ChatAnswer, Citation, DocumentChunk

logger = logging.getLogger(__name__)


def _build_citations(hits: list[tuple[DocumentChunk, float]], max_items: int = 3) -> list[Citation]:
    citations: list[Citation] = []
    for chunk, score in hits[:max_items]:
        excerpt = chunk.text[:220].strip()
        citations.append(
            Citation(
                source_file=chunk.metadata.source_file,
                lecture_id=chunk.metadata.lecture_id,
                lecture_title=chunk.metadata.lecture_title,
                chunk_id=chunk.metadata.chunk_id,
                excerpt=excerpt,
                score=score,
            )
        )
    return citations


def _compose_extractive_answer(question: str, hits: list[tuple[DocumentChunk, float]]) -> str:
    top_chunk = hits[0][0] if hits else None
    top_chunk_text = top_chunk.text if top_chunk else ""
    if len(top_chunk_text) > 360:
        top_chunk_text = top_chunk_text[:360].rstrip() + "..."
    lecture_title = top_chunk.metadata.lecture_title if top_chunk else "unknown lecture"
    return (
        f"Based on the course transcripts, here is the best supported answer to your question "
        f"('{question}') from lecture '{lecture_title}':\n\n{top_chunk_text}"
    )


def _build_context_block(hits: list[tuple[DocumentChunk, float]], max_chunks: int = 8) -> str:
    lines: list[str] = []
    for i, (chunk, score) in enumerate(hits[:max_chunks], start=1):
        lines.append(
            f"[{i}] Lecture: {chunk.metadata.lecture_title} | File: {chunk.metadata.source_file} | "
            f"Chunk: {chunk.metadata.chunk_id} | Score: {score:.3f}\n{chunk.text}"
        )
    return "\n\n".join(lines)


def _compose_generative_answer(
    question: str,
    hits: list[tuple[DocumentChunk, float]],
    settings: Settings,
) -> str | None:
    if not settings.use_llm_grounded_answers:
        return None
    if settings.llm_provider.strip().lower() != "openai":
        return None
    if not settings.openai_api_key.strip():
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    context = _build_context_block(hits, max_chunks=8)
    if not context.strip():
        return None

    system_prompt = (
        "You are a strict course assistant.\n"
        "Answer using only the transcript evidence provided.\n"
        "If evidence is weak or ambiguous, say so briefly.\n"
        "Prefer concise, direct answers.\n"
        "Do not invent facts outside the evidence."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Transcript evidence:\n{context}\n\n"
        "Write a grounded answer in the same language as the question."
    )

    try:
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.openai_model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content if response.choices else ""
        text = (content or "").strip()
        return text or None
    except Exception as exc:  # pragma: no cover - external API/network path
        logger.warning("OpenAI answer generation failed, falling back to extractive answer: %s", exc)
        return None


def generate_grounded_answer(
    question: str,
    hits: list[tuple[DocumentChunk, float]],
    settings: Settings,
) -> ChatAnswer:
    if not hits:
        return ChatAnswer(
            answer=settings.insufficient_evidence_message,
            citations=[],
            grounded=False,
        )

    citations = _build_citations(hits, max_items=5)
    answer = _compose_generative_answer(question=question, hits=hits, settings=settings)
    if not answer:
        answer = _compose_extractive_answer(question, hits)
    return ChatAnswer(answer=answer, citations=citations, grounded=True)
