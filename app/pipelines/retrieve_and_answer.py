from app.core.config import Settings
from app.domain.models import ChatAnswer, ChatTurn
from app.services.answer_service import generate_grounded_answer
from app.services.conversation_context import build_retrieval_question
from app.services.course_router import choose_course_id
from app.services.retrieve_service import retrieve_chunks


def run_retrieve_and_answer(
    question: str,
    settings: Settings,
    course_id: str | None = None,
    chat_history: list[ChatTurn] | None = None,
) -> ChatAnswer:
    history = chat_history or []
    retrieval_question = build_retrieval_question(question=question, chat_history=history)
    selected_course_id = choose_course_id(
        question=retrieval_question,
        settings=settings,
        explicit_course_id=course_id,
    )
    hits = retrieve_chunks(question=retrieval_question, settings=settings, course_id=selected_course_id)
    return generate_grounded_answer(
        question=question,
        hits=hits,
        settings=settings,
        chat_history=history,
    )
