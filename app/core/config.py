from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="Course RAG Chatbot", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    default_course_id: str = Field(default="machine_learning", alias="DEFAULT_COURSE_ID")

    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    media_dir: Path = Field(default=Path("./data/media"), alias="MEDIA_DIR")
    transcripts_dir: Path = Field(default=Path("./data/transcripts"), alias="TRANSCRIPTS_DIR")
    index_dir: Path = Field(default=Path("./data/indexes"), alias="INDEX_DIR")
    metadata_dir: Path = Field(default=Path("./data/metadata"), alias="METADATA_DIR")

    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")
    chunk_quality_filter_enabled: bool = Field(default=True, alias="CHUNK_QUALITY_FILTER_ENABLED")
    min_chunk_alpha_ratio: float = Field(default=0.35, alias="MIN_CHUNK_ALPHA_RATIO")
    max_chunk_digit_ratio: float = Field(default=0.30, alias="MAX_CHUNK_DIGIT_RATIO")
    min_chunk_unique_token_ratio: float = Field(default=0.20, alias="MIN_CHUNK_UNIQUE_TOKEN_RATIO")
    max_chunk_repeated_token_run: int = Field(default=10, alias="MAX_CHUNK_REPEATED_TOKEN_RUN")
    top_k: int = Field(default=6, alias="TOP_K")
    retrieval_pool_k: int = Field(default=36, alias="RETRIEVAL_POOL_K")
    embedding_dim: int = Field(default=256, alias="EMBEDDING_DIM")
    min_retrieval_score: float = Field(default=0.30, alias="MIN_RETRIEVAL_SCORE")
    min_evidence_hits: int = Field(default=1, alias="MIN_EVIDENCE_HITS")
    min_lexical_overlap: float = Field(default=0.12, alias="MIN_LEXICAL_OVERLAP")
    insufficient_evidence_message: str = Field(
        default="I do not have enough evidence from the course transcripts to answer this confidently.",
        alias="INSUFFICIENT_EVIDENCE_MESSAGE",
    )

    transcription_manifest_name: str = Field(default="lectures_manifest.json", alias="TRANSCRIPTION_MANIFEST_NAME")
    transcription_lectures_metadata_name: str = Field(
        default="lectures.json",
        alias="TRANSCRIPTION_LECTURES_METADATA_NAME",
    )

    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    use_llm_grounded_answers: bool = Field(default=False, alias="USE_LLM_GROUNDED_ANSWERS")
    embedding_provider: str = Field(default="local_hash", alias="EMBEDDING_PROVIDER")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="OPENAI_EMBEDDING_MODEL",
    )
    rerank_provider: str = Field(default="none", alias="RERANK_PROVIDER")
    cohere_api_key: str = Field(default="", alias="COHERE_API_KEY")
    cohere_rerank_model: str = Field(
        default="rerank-multilingual-v3.0",
        alias="COHERE_RERANK_MODEL",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
