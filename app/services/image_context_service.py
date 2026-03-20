import base64
import logging
import re

from app.core.config import Settings

logger = logging.getLogger(__name__)
_VISION_FAILURE_PATTERNS = [
    re.compile(r"i (?:cannot|can't|do not|don't) (?:see|view|access) (?:the )?image", re.IGNORECASE),
    re.compile(r"no (?:image|photo|picture) (?:was )?(?:provided|attached)", re.IGNORECASE),
    re.compile(r"אין לי מידע על התמונה"),
]


def _is_vision_failure_text(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return True
    return any(pattern.search(normalized) for pattern in _VISION_FAILURE_PATTERNS)


def build_image_context_for_question(
    question: str,
    image_bytes: bytes,
    mime_type: str,
    settings: Settings,
) -> str:
    if not image_bytes:
        return ""
    if settings.llm_provider.strip().lower() != "openai":
        return ""
    if not settings.openai_api_key.strip():
        return ""

    try:
        from openai import OpenAI
    except ImportError:
        return ""

    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime_type};base64,{image_b64}"

    prompt = (
        "Extract useful learning context from this image for answering a user question.\n"
        "Focus on OCR text, equations, symbols, and diagram labels.\n"
        "Preserve equations in valid LaTeX markdown using $...$ (inline) or $$...$$ (block).\n"
        "If the image contains an example/problem statement, include it explicitly.\n"
        "Return concise structured markdown."
    )

    try:
        client = OpenAI(api_key=settings.openai_api_key)

        # First try the modern Responses API for multimodal reliability.
        try:
            response = client.responses.create(
                model=settings.openai_vision_model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": f"Question: {question}\n\n{prompt}"},
                            {"type": "input_image", "image_url": data_url},
                        ],
                    }
                ],
                temperature=0.0,
            )
            content = (response.output_text or "").strip()
            if not _is_vision_failure_text(content):
                return content
        except Exception as exc:  # pragma: no cover - provider version/compat path
            logger.info("Responses API vision path failed, trying chat.completions fallback: %s", exc)

        # Fallback: chat.completions multimodal format.
        response = client.chat.completions.create(
            model=settings.openai_vision_model,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Question: {question}\n\n{prompt}"},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
        content = response.choices[0].message.content if response.choices else ""
        text = (content or "").strip()
        if _is_vision_failure_text(text):
            return ""
        return text
    except Exception as exc:  # pragma: no cover - external API/network path
        logger.warning("Image context extraction failed: %s", exc)
        return ""
