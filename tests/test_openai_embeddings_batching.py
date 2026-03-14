from types import SimpleNamespace

from app.providers.embeddings.openai_embeddings import OpenAIEmbeddingProvider


class _FakeEmbeddingsApi:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def create(self, model: str, input: list[str]):
        _ = model
        self.calls.append(list(input))
        data = [SimpleNamespace(embedding=[float(i)]) for i, _ in enumerate(input, start=1)]
        return SimpleNamespace(data=data)


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddingsApi()


def test_embed_documents_batches_large_inputs(monkeypatch) -> None:
    def _fake_openai(api_key: str):
        _ = api_key
        return _FakeOpenAIClient()

    monkeypatch.setattr("app.providers.embeddings.openai_embeddings.OpenAI", _fake_openai)
    provider = OpenAIEmbeddingProvider(api_key="k", model="text-embedding-3-large")

    # Force small batch budget to test batching deterministically.
    provider._MAX_TOKENS_PER_REQUEST = 10
    provider._EST_CHARS_PER_TOKEN = 1

    vectors = provider.embed_documents(["abcde", "abcde", "abcde"])

    assert len(vectors) == 3
    assert len(provider._client.embeddings.calls) == 2
    assert provider._client.embeddings.calls[0] == ["abcde", "abcde"]
    assert provider._client.embeddings.calls[1] == ["abcde"]
