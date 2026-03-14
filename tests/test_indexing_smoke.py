from pathlib import Path

import pytest

from app.core.config import Settings
from app.pipelines.ingest_and_index import run_ingest_and_index
from app.providers.embeddings.local_hash_embeddings import LocalHashEmbeddingProvider
from app.providers.vectorstore.faiss_store import FaissVectorStore


faiss = pytest.importorskip("faiss")


def test_ingest_and_faiss_index_smoke(tmp_path: Path) -> None:
    _ = faiss  # keep importorskip explicit for linters

    settings = Settings(
        DEFAULT_COURSE_ID="machine_learning",
        TRANSCRIPTS_DIR=Path("data/transcripts"),
        INDEX_DIR=tmp_path,
        EMBEDDING_PROVIDER="local_hash",
        CHUNK_SIZE=120,
        CHUNK_OVERLAP=20,
        EMBEDDING_DIM=64,
    )

    stats = run_ingest_and_index(settings)
    assert stats["documents"] >= 1
    assert stats["indexed"] >= 1

    store = FaissVectorStore(tmp_path, "machine_learning")
    query_embedding = LocalHashEmbeddingProvider(dim=64).embed_query("What is supervised learning?")
    hits = store.search(query_embedding, top_k=3)

    assert len(hits) >= 1
    top_chunk, score = hits[0]
    assert top_chunk.metadata.course_id == "machine_learning"
    assert score >= -1.0
