import hashlib
import math
import re

from app.domain.interfaces import EmbeddingProvider

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


class LocalHashEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic local embedding provider for MVP development.

    This avoids external API calls while preserving a provider interface that can
    be swapped later with OpenAI or another embedding backend.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("Embedding dimension must be > 0")
        self.dim = dim

    def _tokenize(self, text: str) -> list[str]:
        return _TOKEN_PATTERN.findall(text.lower())

    def _token_to_index_sign(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, byteorder="big", signed=False)
        idx = raw % self.dim
        sign = 1.0 if ((raw >> 1) & 1) == 0 else -1.0
        return idx, sign

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = self._tokenize(text)
        if not tokens:
            return vec

        for token in tokens:
            idx, sign = self._token_to_index_sign(token)
            vec[idx] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
