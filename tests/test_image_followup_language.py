from app.core.config import Settings
from app.services import answer_service


def test_image_followup_uses_language_aware_image_only_generation(monkeypatch) -> None:
    settings = Settings(
        USE_LLM_GROUNDED_ANSWERS=True,
        EMBEDDING_PROVIDER="local_hash",
    )

    def _fake_image_only_answer(question, image_context, settings, response_language, chat_history):
        _ = question
        _ = image_context
        _ = settings
        _ = chat_history
        assert response_language == "he"
        return "זה הסבר בעברית על סמך התמונה."

    monkeypatch.setattr(answer_service, "_compose_image_only_answer", _fake_image_only_answer)

    result = answer_service.generate_grounded_answer(
        question="תסביר בעברית",
        hits=[],
        settings=settings,
        additional_context="Summary in English",
        chat_history=[],
    )
    assert result.grounded is False
    assert result.answer == "זה הסבר בעברית על סמך התמונה."


def test_image_followup_falls_back_to_prefix_when_llm_not_available() -> None:
    settings = Settings(
        USE_LLM_GROUNDED_ANSWERS=False,
        EMBEDDING_PROVIDER="local_hash",
    )

    result = answer_service.generate_grounded_answer(
        question="תסביר בעברית",
        hits=[],
        settings=settings,
        additional_context="Summary in English",
        chat_history=[],
    )
    assert result.grounded is False
    assert result.answer.startswith("בהתבסס על התמונה שהועלתה:")
