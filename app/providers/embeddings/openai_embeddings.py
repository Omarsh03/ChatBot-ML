from openai import BadRequestError, OpenAI

from app.domain.interfaces import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    _MAX_TOKENS_PER_REQUEST = 280_000
    _EST_CHARS_PER_TOKEN = 4

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key.strip():
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // self._EST_CHARS_PER_TOKEN)

    def _chunk_by_estimated_tokens(self, texts: list[str]) -> list[list[str]]:
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_tokens = 0

        for text in texts:
            estimated = self._estimate_tokens(text)

            if current_batch and current_tokens + estimated > self._MAX_TOKENS_PER_REQUEST:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            # If a single text is extremely large, still send it alone.
            current_batch.append(text)
            current_tokens += estimated

        if current_batch:
            batches.append(current_batch)

        return batches

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self._client.embeddings.create(model=self._model, input=texts)
            return [item.embedding for item in response.data]
        except BadRequestError as error:
            message = str(error)
            token_limit_markers = (
                "max_tokens_per_request",
                "maximum request size",
                "maximum request size is 300000 tokens",
            )
            is_token_limit = any(marker in message.lower() for marker in token_limit_markers)
            if not is_token_limit or len(texts) <= 1:
                raise
            middle = max(1, len(texts) // 2)
            left = self._embed_batch(texts[:middle])
            right = self._embed_batch(texts[middle:])
            return left + right

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for batch in self._chunk_by_estimated_tokens(texts):
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self._model, input=[text])
        return response.data[0].embedding
