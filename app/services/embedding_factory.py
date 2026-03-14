import logging

from app.core.config import Settings
from app.domain.interfaces import EmbeddingProvider
from app.providers.embeddings.local_hash_embeddings import LocalHashEmbeddingProvider

logger = logging.getLogger(__name__)


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    provider = settings.embedding_provider.strip().lower()

    if provider in {"local_hash", "local"}:
        return LocalHashEmbeddingProvider(dim=settings.embedding_dim)

    if provider == "openai":
        if not settings.openai_api_key.strip():
            logger.warning(
                "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is missing. "
                "Falling back to local_hash embeddings."
            )
            return LocalHashEmbeddingProvider(dim=settings.embedding_dim)
        from app.providers.embeddings.openai_embeddings import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
        )

    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
