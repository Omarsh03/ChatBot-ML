from app.core.config import Settings
from app.services.answer_service import generate_grounded_answer


def test_hebrew_greeting_returns_smalltalk_response() -> None:
    settings = Settings(USE_LLM_GROUNDED_ANSWERS=False)
    result = generate_grounded_answer(question="היי", hits=[], settings=settings)
    assert result.grounded is False
    assert result.citations == []
    assert "אני כאן כדי לעזור" in result.answer


def test_english_greeting_returns_smalltalk_response() -> None:
    settings = Settings(USE_LLM_GROUNDED_ANSWERS=False)
    result = generate_grounded_answer(question="How are you today?", hits=[], settings=settings)
    assert result.grounded is False
    assert result.citations == []
    assert "I am here to help" in result.answer


def test_non_greeting_without_hits_keeps_insufficient_message() -> None:
    settings = Settings(
        USE_LLM_GROUNDED_ANSWERS=False,
        INSUFFICIENT_EVIDENCE_MESSAGE="INSUFFICIENT",
    )
    result = generate_grounded_answer(question="Explain SVM", hits=[], settings=settings)
    assert result.grounded is False
    assert result.answer == "INSUFFICIENT"
