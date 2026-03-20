# Course RAG Chatbot (MVP)

Python-first MVP for a university course chatbot that answers questions from transcript files.

Current target scope:
- One course only: `machine_learning`
- Local transcript `.txt` ingestion
- Clean architecture seams for chunking/indexing/retrieval/generation
- FastAPI backend + Streamlit UI

## Project status

This repository currently includes **Milestone 1 to Transcription T6**:
- bootstrap scaffold and configuration
- domain models and provider interfaces
- mock transcript files for early development
- local transcript ingestion service
- deterministic chunking service
- local deterministic embedding provider
- FAISS indexing with local persistence
- retrieval from FAISS index
- grounded answer generation with citations
- explicit insufficient-evidence fallback response
- FastAPI endpoints (`/health`, `/ingest`, `/chat`, `/chat_image`)
- transcription integration pipeline using `WhisperHe.transcribe(path)`
- minimal transcription trigger endpoint (`/transcribe`)
- Streamlit chat UI with optional image upload (text-only flow unchanged)
- Milestone smoke tests for ingestion, indexing, retrieval/answering, and API

This MVP now supports end-to-end local demo flow.

## Exact run steps (Windows PowerShell)

For best retrieval quality, configure your own provider API key in `.env` (for example `OPENAI_API_KEY`).
Without a provider API key, the app can still run using local-hash fallback retrieval with lower answer quality.

1. Create and activate a virtual environment (Python 3.13 example):
   - `py -3.13 -m venv .venv`
   - `.venv\Scripts\Activate.ps1`
2. Install base dependencies:
   - `py -3.13 -m pip install -e .`
3. Install transcription engine dependencies:
   - `py -3.13 -m pip install -e .[transcription]`
4. Copy env template:
   - `Copy-Item .env.example .env`
5. Start API:
   - `py -3.13 scripts/run_api.py`
6. Build/update index from local transcripts:
   - `py -3.13 scripts/ingest_local.py`
7. Start Streamlit UI (in a second terminal):
   - `py -3.13 scripts/run_ui.py`
8. Open UI:
   - `http://localhost:8501`

## Transcription pipeline run steps (local media files)

1. Put lecture media files in:
   - `data/media/machine_learning/`
2. Optionally provide manifest file for explicit metadata:
   - `data/metadata/machine_learning/lectures_manifest.json`
   - format:
     - `[{ "media_path": "absolute_or_relative_path", "lecture_id": "lec_01", "lecture_title": "Intro to ML" }]`
3. Run transcription + ingest/index:
   - `py -3.13 scripts/transcribe_local.py`
   - transcription runtime is CUDA-first (`device="cuda"`, `compute_type="float16"`)
   - if CUDA libraries are unavailable, it automatically falls back to CPU (`device="cpu"`, `compute_type="int8"`)
4. Generated outputs:
   - transcripts: `data/transcripts/machine_learning/*.txt`
   - lecture metadata: `data/metadata/machine_learning/lectures.json`
   - index: `data/indexes/machine_learning/`

## API quick checks

- Health:
  - `Invoke-RestMethod -Method GET -Uri "http://127.0.0.1:8000/health"`
- Ingest:
  - `Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:8000/ingest"`
- Transcribe (and ingest/index):
  - `Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:8000/transcribe" -ContentType "application/json" -Body '{"course_id":"machine_learning","run_ingestion":true}'`
- Chat:
  - `Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:8000/chat" -ContentType "application/json" -Body '{"question":"What is supervised learning?"}'`

## Retrieval quality mode (recommended)

To improve answer quality, switch from local hash embeddings to OpenAI embeddings and optional Cohere reranking:

- `.env` settings:
  - `USE_LLM_GROUNDED_ANSWERS=true`
  - `EMBEDDING_PROVIDER=openai`
  - `OPENAI_API_KEY=<your_key>`
  - `OPENAI_EMBEDDING_MODEL=text-embedding-3-large`
  - `OPENAI_VISION_MODEL=gpt-4o-mini` (used when a user uploads an image)
  - `RERANK_PROVIDER=cohere`
  - `COHERE_API_KEY=<your_key>`
  - `COHERE_RERANK_MODEL=rerank-multilingual-v3.0`
  - `RETRIEVAL_POOL_K=36`
  - `TOP_K=6`
  - `MIN_RETRIEVAL_SCORE=0.30`
  - `MIN_LEXICAL_OVERLAP=0.12`
  - `PROBABILITY_RETRIEVAL_POOL_K=48`
  - `PROBABILITY_MIN_RETRIEVAL_SCORE=0.22`
  - `PROBABILITY_MIN_LEXICAL_OVERLAP=0.08`
  - `PROBABILITY_MIN_EVIDENCE_HITS=1`
  - `CHUNK_QUALITY_FILTER_ENABLED=true`
  - `MIN_CHUNK_ALPHA_RATIO=0.35`
  - `MAX_CHUNK_DIGIT_RATIO=0.30`
  - `MIN_CHUNK_UNIQUE_TOKEN_RATIO=0.20`
  - `MAX_CHUNK_REPEATED_TOKEN_RUN=10`
- Rebuild the index after changing embedding provider/model:
  - `py -3.13 scripts/ingest_local.py`

## Run tests

- All tests:
  - `py -3.13 -m pytest -q`

## Grounded-answer policy

The design includes a strict fallback rule:
- if retrieved transcript evidence is insufficient, the chatbot must explicitly say it does not have enough evidence from transcripts.

## Transcript metadata schema (from day one)

Each chunk is designed to carry:
- `course_id`
- `lecture_id`
- `lecture_title`
- `source_file`
- `chunk_id`
- optional future fields: `start_ts`, `end_ts`

## Retrieval grounding guardrails

To reduce false matches from generic chunks:
- chunks are embedded with lecture title context (`Lecture: <title> + chunk_text`)
- retrieval requires minimum vector score
- retrieval requires lexical overlap between question and evidence
- retrieval can require minimum number of supporting hits
- if these checks fail, chatbot returns the insufficient-evidence response
