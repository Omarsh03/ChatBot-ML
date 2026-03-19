import re

from app.domain.models import ChatTurn

_SHORT_FOLLOWUP_KEYWORDS = (
    "קצר",
    "בקצרה",
    "תמצית",
    "תסכם",
    "brief",
    "short",
    "shorter",
    "summarize",
    "summary",
)
_TRANSLATION_OR_LANGUAGE_KEYWORDS = (
    "באנגלית",
    "בעברית",
    "english",
    "hebrew",
    "translate",
    "תרגם",
    "תתרגם",
)
_EXAMPLE_FOLLOWUP_KEYWORDS = (
    "דוגמה נוספת",
    "עוד דוגמה",
    "another example",
    "one more example",
)
_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def _last_user_question(chat_history: list[ChatTurn]) -> str | None:
    for turn in reversed(chat_history):
        if turn.role == "user" and turn.content.strip():
            return turn.content.strip()
    return None


def is_contextual_followup(question: str) -> bool:
    text = question.strip().lower()
    if not text:
        return False

    keyword_groups = (
        _SHORT_FOLLOWUP_KEYWORDS,
        _TRANSLATION_OR_LANGUAGE_KEYWORDS,
        _EXAMPLE_FOLLOWUP_KEYWORDS,
    )
    if any(keyword in text for group in keyword_groups for keyword in group):
        return True

    # Very short prompts are often follow-ups ("short", "why?", "in english", etc.)
    token_count = len(_TOKEN_PATTERN.findall(text))
    return token_count <= 3


def build_retrieval_question(question: str, chat_history: list[ChatTurn]) -> str:
    if not is_contextual_followup(question):
        return question
    last_question = _last_user_question(chat_history)
    if not last_question:
        return question
    return f"Previous user question: {last_question}\nFollow-up instruction: {question}"


def is_brief_requested(question: str) -> bool:
    text = question.strip().lower()
    return any(keyword in text for keyword in _SHORT_FOLLOWUP_KEYWORDS)
