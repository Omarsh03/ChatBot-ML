import logging
import re
import base64

from app.core.config import Settings
from app.domain.models import ChatAnswer, ChatTurn, Citation, DocumentChunk
from app.services.conversation_context import is_brief_requested

logger = logging.getLogger(__name__)

_HE_WORD_PATTERN = re.compile(r"[\u0590-\u05FF]+")
_EN_WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_GREETING_PATTERNS_HE = [
    re.compile(r"^\s*ה(?:י|יי)\s*[!.?]*\s*$"),
    re.compile(r"^\s*שלום(?:\s+לך|\s+לכם)?\s*[!.?]*\s*$"),
    re.compile(r"מה\s+קורה"),
    re.compile(r"מה\s+הולך"),
    re.compile(r"איך\s+אתה\s+היום"),
    re.compile(r"איך\s+את\s+היום"),
    re.compile(r"מה\s+שלומך"),
]
_GREETING_PATTERNS_EN = [
    re.compile(r"^\s*hi+\s*[!.?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*hello+\s*[!.?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*hey+\s*[!.?]*\s*$", re.IGNORECASE),
    re.compile(r"\bhow are you(?: today)?\b", re.IGNORECASE),
    re.compile(r"\bwhat'?s up\b", re.IGNORECASE),
    re.compile(r"\bhow'?s it going\b", re.IGNORECASE),
]
_LANG_OVERRIDE_INDICATORS: list[tuple[str, re.Pattern[str]]] = [
    ("en", re.compile(r"\benglish\b", re.IGNORECASE)),
    ("en", re.compile(r"באנגלית")),
    ("he", re.compile(r"\bhebrew\b", re.IGNORECASE)),
    ("he", re.compile(r"בעברית")),
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
    for lang, pattern in _LANG_OVERRIDE_INDICATORS:
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


def _is_greeting_or_smalltalk(question: str) -> bool:
    normalized = question.strip().lower()
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in _GREETING_PATTERNS_HE + _GREETING_PATTERNS_EN)


def _smalltalk_response(language: str) -> str:
    if language == "he":
        return "היי! אני כאן כדי לעזור לך בשאלות על חומר הלימוד. אפשר לשאול כל שאלה."
    return "Hi! I am here to help with course-related questions. Feel free to ask anything."


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
    chat_history: list[ChatTurn],
    additional_context: str = "",
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

    recent_history = chat_history[-6:]
    history_block = "\n".join(f"{turn.role}: {turn.content}" for turn in recent_history if turn.content.strip())
    brevity_instruction = (
        "Keep the answer very concise (3-5 lines max)." if is_brief_requested(question) else "Keep the answer concise."
    )

    system_prompt = (
        "You are a strict course assistant.\n"
        "Answer using only the transcript evidence provided.\n"
        "Treat prior image discussion as optional context: use it only if the current question clearly refers to it.\n"
        "If evidence is weak or ambiguous, say so briefly.\n"
        f"{brevity_instruction}\n"
        "Do not invent facts outside the evidence.\n"
        "When writing math, format equations in LaTeX markdown using $...$ for inline and $$...$$ for blocks."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"User-provided image context:\n{additional_context or '[none]'}\n\n"
        f"Recent conversation:\n{history_block or '[none]'}\n\n"
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


def _compose_multimodal_answer(
    question: str,
    hits: list[tuple[DocumentChunk, float]],
    settings: Settings,
    response_language: str,
    chat_history: list[ChatTurn],
    image_bytes: bytes,
    mime_type: str,
    additional_context: str = "",
) -> str | None:
    if not settings.use_llm_grounded_answers:
        return None
    if settings.llm_provider.strip().lower() != "openai":
        return None
    if not settings.openai_api_key.strip():
        return None
    if not image_bytes:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    context = _build_context_block(hits, max_chunks=8)
    if not context.strip():
        return None

    recent_history = chat_history[-6:]
    history_block = "\n".join(f"{turn.role}: {turn.content}" for turn in recent_history if turn.content.strip())
    brevity_instruction = (
        "Keep the answer very concise (3-5 lines max)." if is_brief_requested(question) else "Keep the answer concise."
    )
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime_type};base64,{image_b64}"

    system_prompt = (
        "You are a strict multimodal course assistant.\n"
        "Use both: (1) transcript evidence and (2) the uploaded image.\n"
        "Prioritize transcript evidence for factual grounding, and use image details to interpret formulas/diagrams.\n"
        "If the image and transcripts conflict, explicitly mention the mismatch briefly.\n"
        f"{brevity_instruction}\n"
        "Do not invent facts outside evidence.\n"
        "When writing math, format equations in LaTeX markdown using $...$ for inline and $$...$$ for blocks."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"User-provided image context:\n{additional_context or '[none]'}\n\n"
        f"Recent conversation:\n{history_block or '[none]'}\n\n"
        f"Transcript evidence:\n{context}\n\n"
        f"Write a grounded answer in {'Hebrew' if response_language == 'he' else 'English'}."
    )

    try:
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.openai_vision_model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )
        content = response.choices[0].message.content if response.choices else ""
        text = (content or "").strip()
        return text or None
    except Exception as exc:  # pragma: no cover - external API/network path
        logger.warning("OpenAI multimodal answer generation failed, falling back to text-only answer: %s", exc)
        return None


def _compose_image_only_answer(
    question: str,
    image_context: str,
    settings: Settings,
    response_language: str,
    chat_history: list[ChatTurn],
) -> str | None:
    if not settings.use_llm_grounded_answers:
        return None
    if settings.llm_provider.strip().lower() != "openai":
        return None
    if not settings.openai_api_key.strip():
        return None
    if not image_context.strip():
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    recent_history = chat_history[-6:]
    history_block = "\n".join(f"{turn.role}: {turn.content}" for turn in recent_history if turn.content.strip())
    target_lang = "Hebrew" if response_language == "he" else "English"
    system_prompt = (
        "You are a strict course assistant.\n"
        "The user asked about an uploaded image.\n"
        "Rewrite/explain the provided image context clearly in the user's requested language.\n"
        "Do not claim transcript evidence if none was provided.\n"
        "When writing math, format equations in LaTeX markdown using $...$ for inline and $$...$$ for blocks."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Recent conversation:\n{history_block or '[none]'}\n\n"
        f"Image context:\n{image_context}\n\n"
        f"Write the answer in {target_lang}. Keep it concise and accurate."
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
        logger.warning("OpenAI image-only answer generation failed: %s", exc)
        return None


def generate_grounded_answer(
    question: str,
    hits: list[tuple[DocumentChunk, float]],
    settings: Settings,
    chat_history: list[ChatTurn] | None = None,
    additional_context: str = "",
    image_bytes: bytes = b"",
    image_mime_type: str = "image/png",
) -> ChatAnswer:
    history = chat_history or []
    response_language = _determine_response_language(question)

    if _is_greeting_or_smalltalk(question):
        return ChatAnswer(
            answer=_smalltalk_response(response_language),
            citations=[],
            grounded=False,
        )

    if not hits:
        if additional_context.strip():
            image_only_answer = _compose_image_only_answer(
                question=question,
                image_context=additional_context.strip(),
                settings=settings,
                response_language=response_language,
                chat_history=history,
            )
            if image_only_answer:
                return ChatAnswer(
                    answer=image_only_answer,
                    citations=[],
                    grounded=False,
                )
            prefix = "Based on the uploaded image" if response_language == "en" else "בהתבסס על התמונה שהועלתה"
            return ChatAnswer(
                answer=f"{prefix}:\n\n{additional_context.strip()}",
                citations=[],
                grounded=False,
            )
        return ChatAnswer(
            answer=settings.insufficient_evidence_message,
            citations=[],
            grounded=False,
        )

    citations = _build_citations(hits, max_items=5)
    answer = _compose_multimodal_answer(
        question=question,
        hits=hits,
        settings=settings,
        response_language=response_language,
        chat_history=history,
        image_bytes=image_bytes,
        mime_type=image_mime_type,
        additional_context=additional_context,
    )
    if not answer:
        answer = _compose_generative_answer(
            question=question,
            hits=hits,
            settings=settings,
            response_language=response_language,
            chat_history=history,
            additional_context=additional_context,
        )
    if not answer:
        answer = _compose_extractive_answer(question, hits, response_language=response_language)
    return ChatAnswer(answer=answer, citations=citations, grounded=True)
