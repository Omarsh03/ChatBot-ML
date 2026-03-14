from app.core.config import Settings
from app.domain.models import ChatAnswer
from app.services.answer_service import generate_grounded_answer
from app.services.retrieve_service import retrieve_chunks


def run_retrieve_and_answer(
    question: str,
    settings: Settings,
    course_id: str | None = None,
) -> ChatAnswer:
    hits = retrieve_chunks(question=question, settings=settings, course_id=course_id)
    return generate_grounded_answer(question=question, hits=hits, settings=settings)
