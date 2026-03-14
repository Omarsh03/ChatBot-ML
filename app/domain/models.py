from typing import Optional

from pydantic import BaseModel, Field


class TranscriptChunkMetadata(BaseModel):
    course_id: str = Field(..., description="Course identifier, e.g. machine_learning")
    lecture_id: str = Field(..., description="Lecture identifier, e.g. lec_01")
    lecture_title: str = Field(..., description="Human-readable lecture title")
    source_file: str = Field(..., description="Original transcript file name")
    chunk_id: str = Field(..., description="Stable chunk identifier")
    start_ts: Optional[float] = Field(default=None, description="Optional start timestamp in seconds")
    end_ts: Optional[float] = Field(default=None, description="Optional end timestamp in seconds")


class DocumentChunk(BaseModel):
    text: str
    metadata: TranscriptChunkMetadata


class TranscriptDocument(BaseModel):
    course_id: str
    lecture_id: str
    lecture_title: str
    source_file: str
    text: str


class Citation(BaseModel):
    source_file: str
    lecture_id: str
    lecture_title: str
    chunk_id: str
    excerpt: str
    score: Optional[float] = None


class ChatRequest(BaseModel):
    question: str
    course_id: str = "machine_learning"


class IngestRequest(BaseModel):
    course_id: str = "machine_learning"


class TranscribeRequest(BaseModel):
    course_id: str = "machine_learning"
    run_ingestion: bool = True


class ChatAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = True
