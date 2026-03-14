import logging
import re

from app.core.config import Settings
from app.domain.models import ChatAnswer, Citation, DocumentChunk

logger = logging.getLogger(__name__)

_HE_WORD_PATTERN = re.compile(r"[\u0590-\u05FF]+")
_EN_WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_LANG_OVERRIDE_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("en", re.compile(r"\bin english\b", re.IGNORECASE)),
    ("en", re.compile(r"\b(answer|reply|respond|write|explain)\s+(in\s+)?english\b", re.IGNORECASE)),
    ("en", re.compile(r"תענ[הי]\s+באנגלית")),
    ("en", re.compile(r"ת(?:ן|ני)\s+לי\s+תשובה\s+באנגלית")),
    ("he", re.compile(r"\bin hebrew\b", re.IGNORECASE)),
    ("he", re.compile(r"\b(answer|reply|respond|write|explain)\s+(in\s+)?hebrew\b", re.IGNORECASE)),
    ("he", re.compile(r"תענ[הי]\s+בעברית")),
    ("he", re.compile(r"ת(?:ן|ני)\s+לי\s+תשובה\s+בעברית")),
]


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


def _explicit_language_override(question: str) -> str | None:
    selected: tuple[str, int] | None = None
    for lang, pattern in _LANG_OVERRIDE_RULES:
        for match in pattern.finditer(question):
            if selected is None or match.start() > selected[1]:
                selected = (lang, match.start())
    return selected[0] if selected else None


def _dominant_language(question: str) -> str:
    he_words = len(_HE_WORD_PATTERN.findall(question))
    en_words = len(_EN_WORD_PATTERN.findall(question))
    if en_words > he_words:
        return "en"
    return "he"


def _determine_response_language(question: str) -> str:
    return _explicit_language_override(question) or _dominant_language(question)


def _compose_extractive_answer(
    question: str,
    hits: list[tuple[DocumentChunk, float]],
    response_language: str,
) -> str:
    top_chunk = hits[0][0] if hits else None
    top_chunk_text = top_chunk.text if top_chunk else ""
    if len(top_chunk_text) > 360:
        top_chunk_text = top_chunk_text[:360].rstrip() + "..."
    lecture_title = top_chunk.metadata.lecture_title if top_chunk else "unknown lecture"
    if response_language == "he":
        return (
            f"בהתבסס על תמלולי הקורס, זו התשובה הנתמכת ביותר לשאלה שלך "
            f"('{question}') מתוך ההרצאה '{lecture_title}':\n\n{top_chunk_text}"
        )
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
    response_language: str,
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
        f"Write a grounded answer in {'Hebrew' if response_language == 'he' else 'English'}."
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
    response_language = _determine_response_language(question)
    answer = _compose_generative_answer(
        question=question,
        hits=hits,
        settings=settings,
        response_language=response_language,
    )
    if not answer:
        answer = _compose_extractive_answer(question, hits, response_language=response_language)
    return ChatAnswer(answer=answer, citations=citations, grounded=True)
