from app.services.answer_service import _determine_response_language


def test_dominant_language_prefers_hebrew_on_mixed_question() -> None:
    question = "אתה יכול להציג לי שאלה של logistic regression ולפתור אותה?"
    assert _determine_response_language(question) == "he"


def test_explicit_english_override_beats_dominant_hebrew() -> None:
    question = "אתה יכול להציג לי שאלה של logistic regression ולפתור אותה? תענה באנגלית"
    assert _determine_response_language(question) == "en"


def test_explicit_hebrew_override_beats_dominant_english() -> None:
    question = "Can you solve a logistic regression question? תענה בעברית"
    assert _determine_response_language(question) == "he"


def test_english_question_defaults_to_english() -> None:
    question = "Can you explain reinforcement learning in simple terms?"
    assert _determine_response_language(question) == "en"
