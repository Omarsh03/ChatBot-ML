from app.domain.interfaces import EmbeddingProvider, VectorStore
from app.domain.models import DocumentChunk


def build_vectors(chunks: list[DocumentChunk], embedding_provider: EmbeddingProvider) -> list[list[float]]:
    texts = [f"Lecture: {c.metadata.lecture_title}\n{c.text}" for c in chunks]
    return embedding_provider.embed_documents(texts)


def index_chunks(
    chunks: list[DocumentChunk],
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
) -> int:
    if not chunks:
        return 0
    vectors = build_vectors(chunks, embedding_provider)
    vector_store.index(chunks, vectors)
    return len(chunks)
