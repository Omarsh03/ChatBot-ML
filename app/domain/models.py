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


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    course_id: Optional[str] = Field(
        default=None,
        description="Optional explicit course ID. If omitted, the backend auto-routes by question.",
    )
    chat_history: list[ChatTurn] = Field(
        default_factory=list,
        description="Optional previous turns for contextual follow-up handling.",
    )
    image_context: str = Field(
        default="",
        description="Optional carried image context for follow-up turns in the same chat session.",
    )


class IngestRequest(BaseModel):
    course_id: str = "machine_learning"


class TranscribeRequest(BaseModel):
    course_id: str = "machine_learning"
    run_ingestion: bool = True


class ChatAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = True
    image_context: str = Field(default="", description="Image context used for this answer, if any.")
