from app.core.config import Settings
from app.domain.models import DocumentChunk
from app.providers.vectorstore.faiss_store import FaissVectorStore
from app.services.embedding_factory import build_embedding_provider
import re

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "have",
    "what",
    "when",
    "where",
    "which",
    "how",
    "who",
    "why",
    "are",
    "was",
    "were",
    "is",
    "in",
    "on",
    "to",
    "of",
    "a",
    "an",
}


def _tokenize(text: str) -> set[str]:
    return {
        t
        for t in _TOKEN_PATTERN.findall(text.lower())
        if len(t) > 2 and t not in _STOPWORDS
    }


def _lexical_overlap(query: str, chunk: DocumentChunk) -> float:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0
    evidence_tokens = _tokenize(f"{chunk.metadata.lecture_title} {chunk.text}")
    if not evidence_tokens:
        return 0.0
    overlap = q_tokens.intersection(evidence_tokens)
    return len(overlap) / len(q_tokens)


def _title_overlap(query: str, chunk: DocumentChunk) -> float:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0
    title_tokens = _tokenize(chunk.metadata.lecture_title)
    if not title_tokens:
        return 0.0
    return len(q_tokens.intersection(title_tokens)) / len(q_tokens)


def retrieve_chunks(
    question: str,
    settings: Settings,
    course_id: str | None = None,
) -> list[tuple[DocumentChunk, float]]:
    selected_course_id = course_id or settings.default_course_id

    embedding_provider = build_embedding_provider(settings)
    vector_store = FaissVectorStore(settings.index_dir, selected_course_id)

    query_vector = embedding_provider.embed_query(question)
    hits = vector_store.search(query_vector, top_k=settings.retrieval_pool_k)

    # Keep only evidence that clears both vector relevance and lexical grounding checks.
    filtered_hits: list[tuple[DocumentChunk, float]] = []
    for chunk, score in hits:
        if score < settings.min_retrieval_score:
            continue
        if _lexical_overlap(question, chunk) < settings.min_lexical_overlap:
            continue
        boosted_score = score + (0.08 * _title_overlap(question, chunk))
        filtered_hits.append((chunk, boosted_score))

    filtered_hits.sort(key=lambda pair: pair[1], reverse=True)

    reranked_hits = _rerank_hits(question=question, hits=filtered_hits, settings=settings)
    final_hits = reranked_hits[: settings.top_k]

    if len(final_hits) < settings.min_evidence_hits:
        return []
    return final_hits


def _rerank_hits(
    question: str,
    hits: list[tuple[DocumentChunk, float]],
    settings: Settings,
) -> list[tuple[DocumentChunk, float]]:
    provider = settings.rerank_provider.strip().lower()
    if provider in {"none", ""} or not hits:
        return hits
    if provider != "cohere":
        return hits
    if not settings.cohere_api_key.strip():
        return hits

    try:
        import cohere
    except ImportError:
        return hits

    try:
        client = cohere.ClientV2(api_key=settings.cohere_api_key)
        documents = [f"{chunk.metadata.lecture_title}\n{chunk.text}" for chunk, _ in hits]
        response = client.rerank(
            model=settings.cohere_rerank_model,
            query=question,
            documents=documents,
            top_n=min(settings.retrieval_pool_k, len(documents)),
        )
        reranked: list[tuple[DocumentChunk, float]] = []
        for row in response.results:
            idx = row.index
            if idx < 0 or idx >= len(hits):
                continue
            chunk, _ = hits[idx]
            reranked.append((chunk, float(row.relevance_score)))
        if reranked:
            return reranked
    except Exception:
        return hits

    return hits
