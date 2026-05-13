# Project Structure (For Now)

```
rag_app/
│
├── main.py                    # FastAPI app entry point
│
├── api/
│   ├── __init__.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── ingest.py          # Document upload & indexing endpoints
│   │   ├── query.py           # RAG query endpoints
│   │   └── health.py          # Health check
│   └── dependencies.py        # Shared FastAPI dependencies (auth, DB sessions)
│
├── core/
│   ├── __init__.py
│   ├── config.py              # Settings (env vars, model names, thresholds)
│   └── logging.py             # Logging setup
│
├── rag/
│   ├── __init__.py
│   ├── pipeline.py            # Orchestrates the full RAG flow
│   ├── retriever.py           # Vector DB search logic
│   ├── generator.py           # LLM call + prompt construction
│   ├── reranker.py            # Optional: cross-encoder reranking
│   └── language_detector.py   # Your Module 1 — lang detection lives here
│
├── ingestion/
│   ├── __init__.py
│   ├── loader.py              # File parsers (PDF, DOCX, TXT...)
│   ├── chunker.py             # Text splitting strategies
│   └── embedder.py            # Embedding model wrapper
│
├── db/
│   ├── __init__.py
│   ├── vector_store.py        # Vector DB client (Chroma, Qdrant, Pinecone...)
│   └── metadata_store.py      # Optional SQL store for doc metadata
│
├── models/
│   ├── __init__.py
│   ├── request.py             # Pydantic input schemas
│   └── response.py            # Pydantic output schemas
│
├── tests/
│   ├── test_pipeline.py
│   ├── test_retriever.py
│   └── test_language_detector.py
│
├── .env
├── requirements.txt
└── Dockerfile
```