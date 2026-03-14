from abc import ABC, abstractmethod

from app.domain.models import ChatAnswer, DocumentChunk


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class ChatProvider(ABC):
    @abstractmethod
    def generate_answer(self, question: str, retrieved_chunks: list[DocumentChunk]) -> ChatAnswer:
        raise NotImplementedError


class VectorStore(ABC):
    @abstractmethod
    def index(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int) -> list[tuple[DocumentChunk, float]]:
        raise NotImplementedError
