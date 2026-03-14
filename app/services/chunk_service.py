from app.domain import build_chunk_id
from app.domain.models import DocumentChunk, TranscriptChunkMetadata, TranscriptDocument
import re

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _windowed_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    normalized = _normalize_whitespace(text)
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    step = chunk_size - chunk_overlap
    n = len(normalized)

    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # Prefer ending on whitespace to avoid splitting words.
            ws = normalized.rfind(" ", start, end)
            if ws > start:
                end = ws
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start += step
    return chunks


def _max_repeated_token_run(tokens: list[str]) -> int:
    if not tokens:
        return 0
    best = 1
    run = 1
    prev = tokens[0]
    for token in tokens[1:]:
        if token == prev:
            run += 1
            best = max(best, run)
        else:
            run = 1
            prev = token
    return best


def _is_chunk_quality_ok(
    text: str,
    *,
    min_alpha_ratio: float,
    max_digit_ratio: float,
    min_unique_token_ratio: float,
    max_repeated_token_run: int,
) -> bool:
    compact = [c for c in text if not c.isspace()]
    if not compact:
        return False

    alpha_count = sum(1 for c in compact if c.isalpha())
    digit_count = sum(1 for c in compact if c.isdigit())
    alpha_ratio = alpha_count / len(compact)
    digit_ratio = digit_count / len(compact)

    if alpha_ratio < min_alpha_ratio:
        return False
    if digit_ratio > max_digit_ratio:
        return False

    tokens = _TOKEN_PATTERN.findall(text.lower())
    if len(tokens) < 8:
        return True

    unique_ratio = len(set(tokens)) / len(tokens)
    if unique_ratio < min_unique_token_ratio:
        return False
    if _max_repeated_token_run(tokens) > max_repeated_token_run:
        return False

    return True


def chunk_transcript_document(
    document: TranscriptDocument,
    chunk_size: int,
    chunk_overlap: int,
    quality_filter_enabled: bool = True,
    min_alpha_ratio: float = 0.35,
    max_digit_ratio: float = 0.30,
    min_unique_token_ratio: float = 0.20,
    max_repeated_token_run: int = 10,
) -> list[DocumentChunk]:
    text_chunks = _windowed_chunks(
        text=document.text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunked_documents: list[DocumentChunk] = []
    for idx, text in enumerate(text_chunks, start=1):
        if quality_filter_enabled and not _is_chunk_quality_ok(
            text,
            min_alpha_ratio=min_alpha_ratio,
            max_digit_ratio=max_digit_ratio,
            min_unique_token_ratio=min_unique_token_ratio,
            max_repeated_token_run=max_repeated_token_run,
        ):
            continue
        metadata = TranscriptChunkMetadata(
            course_id=document.course_id,
            lecture_id=document.lecture_id,
            lecture_title=document.lecture_title,
            source_file=document.source_file,
            chunk_id=build_chunk_id(document.lecture_id, idx),
            start_ts=None,
            end_ts=None,
        )
        chunked_documents.append(DocumentChunk(text=text, metadata=metadata))
    return chunked_documents


def chunk_documents(
    documents: list[TranscriptDocument],
    chunk_size: int,
    chunk_overlap: int,
    quality_filter_enabled: bool = True,
    min_alpha_ratio: float = 0.35,
    max_digit_ratio: float = 0.30,
    min_unique_token_ratio: float = 0.20,
    max_repeated_token_run: int = 10,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for document in documents:
        chunks.extend(
            chunk_transcript_document(
                document=document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                quality_filter_enabled=quality_filter_enabled,
                min_alpha_ratio=min_alpha_ratio,
                max_digit_ratio=max_digit_ratio,
                min_unique_token_ratio=min_unique_token_ratio,
                max_repeated_token_run=max_repeated_token_run,
            )
        )
    return chunks
