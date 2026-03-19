from app.domain.models import ChatTurn
from app.services.conversation_context import (
    build_retrieval_question,
    is_brief_requested,
    is_contextual_followup,
)


def test_short_followup_is_detected() -> None:
    assert is_contextual_followup("קצר")
    assert is_contextual_followup("answer in English")


def test_non_followup_question_not_detected() -> None:
    assert not is_contextual_followup("Explain logistic regression with an example")


def test_retrieval_question_expands_from_history_for_followup() -> None:
    history = [
        ChatTurn(role="user", content="תסביר את חוק בייס"),
        ChatTurn(role="assistant", content="..."),
    ]
    expanded = build_retrieval_question("קצר", history)
    assert "Previous user question: תסביר את חוק בייס" in expanded
    assert "Follow-up instruction: קצר" in expanded


def test_brief_requested_detection() -> None:
    assert is_brief_requested("בקצרה בבקשה")
    assert is_brief_requested("Give a short answer")
    assert not is_brief_requested("Explain with details")
