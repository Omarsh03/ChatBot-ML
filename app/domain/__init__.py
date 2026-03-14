from app.domain.models import TranscriptChunkMetadata


def build_chunk_id(lecture_id: str, index: int) -> str:
    return f"{lecture_id}_chunk_{index:04d}"


__all__ = ["TranscriptChunkMetadata", "build_chunk_id"]
