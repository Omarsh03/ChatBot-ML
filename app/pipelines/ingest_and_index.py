from app.core.config import Settings
from app.providers.vectorstore.faiss_store import FaissVectorStore
from app.services.chunk_service import chunk_documents
from app.services.embedding_factory import build_embedding_provider
from app.services.index_service import index_chunks
from app.services.ingest_service import load_course_transcripts


def run_ingest_and_index(settings: Settings, course_id: str | None = None) -> dict[str, int]:
    selected_course_id = course_id or settings.default_course_id
    docs = load_course_transcripts(settings.transcripts_dir, selected_course_id, settings.metadata_dir)
    chunks = chunk_documents(
        documents=docs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        quality_filter_enabled=settings.chunk_quality_filter_enabled,
        min_alpha_ratio=settings.min_chunk_alpha_ratio,
        max_digit_ratio=settings.max_chunk_digit_ratio,
        min_unique_token_ratio=settings.min_chunk_unique_token_ratio,
        max_repeated_token_run=settings.max_chunk_repeated_token_run,
    )

    embedder = build_embedding_provider(settings)
    vector_store = FaissVectorStore(settings.index_dir, selected_course_id)
    indexed_count = index_chunks(
        chunks=chunks,
        embedding_provider=embedder,
        vector_store=vector_store,
    )

    return {
        "documents": len(docs),
        "chunks": len(chunks),
        "indexed": indexed_count,
    }
