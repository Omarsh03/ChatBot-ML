from pathlib import Path
import json

import numpy as np

from app.domain.interfaces import VectorStore
from app.domain.models import DocumentChunk

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


class FaissVectorStore(VectorStore):
    def __init__(self, base_index_dir: Path, course_id: str) -> None:
        self.course_id = course_id
        self.course_index_dir = base_index_dir / course_id
        self.course_index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.course_index_dir / "index.faiss"
        self.chunks_path = self.course_index_dir / "chunks.jsonl"

    def index(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        if faiss is None:
            raise RuntimeError("faiss is not installed. Install faiss-cpu to use FAISS indexing.")
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        if not chunks:
            raise ValueError("Cannot index empty chunks list")

        matrix = np.asarray(vectors, dtype="float32")
        if matrix.ndim != 2:
            raise ValueError("vectors must be a 2D matrix")
        faiss.normalize_L2(matrix)

        dim = matrix.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)
        faiss.write_index(index, str(self.index_path))

        with self.chunks_path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[DocumentChunk, float]]:
        if faiss is None:
            raise RuntimeError("faiss is not installed. Install faiss-cpu to use FAISS search.")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not self.index_path.exists() or not self.chunks_path.exists():
            return []

        index = faiss.read_index(str(self.index_path))
        chunks = self._load_chunks()
        if not chunks:
            return []

        query = np.asarray([query_vector], dtype="float32")
        faiss.normalize_L2(query)
        scores, indices = index.search(query, top_k)

        results: list[tuple[DocumentChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            results.append((chunks[idx], float(score)))
        return results

    def _load_chunks(self) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        with self.chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunks.append(DocumentChunk.model_validate_json(line))
        return chunks
