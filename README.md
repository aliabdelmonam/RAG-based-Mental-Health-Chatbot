# RAG-Based Mental Health Chatbot

This repository contains a Retrieval-Augmented Generation (RAG) chatbot application designed to provide multilingual mental health support and emotional assistance. The system integrates language detection, intent classification, query rewriting, tool use (such as crisis intervention), and automatic tracing/monitoring.

## Features

- **Multi-Provider Support**: Integrates with Cohere, Gemini, Groq, HuggingFace, OpenAI, and custom Google Colab endpoints.
- **Failover Architecture**: The application employs a primary/fallback provider execution model managed by a central BundleManager. If the primary generation or embedding client fails, the system automatically routes to the fallback client after retrying.
- **Stateless Language Detection**: Integrates a pre-loaded local machine learning model to classify user language, preventing redundant disk-based IO.
- **Intent-Based Routing**: Classifies user messages into greeting, gratitude, crisis, mental health inquiry, or out-of-scope categories, applying distinct pipeline routes (such as direct chatbot responses or retrieval workflows) accordingly.
- **Query Rewriting**: Expands and rewrites incoming queries in context with the recent chat history to optimize retrieval from the vector database.
- **Crisis Intervention Tooling**: Detects crisis situations and automatically invokes specialized tools to provide immediate, curated resources and guidance.
- **LangSmith/LangChain Integration**: Automatic performance monitoring, latency, cost tracking, and execution tracing for all LLM prompts, retrievals, and rewrites.
- **Evaluation Workflow**: Utilizes the Ragas framework inside LlamaIndex Workflows to perform concurrent, batch-controlled evaluations for faithfulness, answer relevancy, and context precision.

## Project Structure

- `src/`
  - `main.py`: Entry point for the FastAPI application.
  - `core/`: Core settings, configuration loading, and logger initialization.
  - `db/`: Vector database interfaces and providers (such as Qdrant).
  - `rag/`: Core modular system containing pipeline orchestrations, conversation histories, intent classifiers, and language detectors.
  - `stores/`: LLM client interfaces, provider factories, schemas, and provider-specific clients.
- `eval/`
  - `eval.py`: Ragas execution script wrapped in LlamaIndex Workflows.
  - `dataset/`: Ground truth evaluation datasets.
- `docker/`: Docker containerization configuration.

## Prerequisites and Installation

### Dependencies
This project uses Poetry for dependency management. Ensure Python 3.12 or 3.13 is installed on your system.

To install dependencies, run:
```bash
poetry install
```

Alternatively, if running via raw Python, ensure that the dependencies specified in `pyproject.toml` are installed:
```bash
pip install -r requirements.txt
```

### Configuration
Create a `.env` file inside the `src/` directory. Use the following environment variables:

```env
APP_ENV=development

# LLM Providers API Keys
GROQ_API_KEY=your_groq_api_key
HF_API_KEY=your_huggingface_api_key
GEMINI_API_KEY=your_gemini_api_key
COHERE_API_KEY=your_cohere_api_key

# Primary LLM and Embedding Providers Configuration
GENERATION_BACKEND_PRIMAY=COHERE
GENERATION_MODEL_ID_PRIMARY=command-a-03-2025
EMBEDDING_BACKEND_PRIMARY=HUGGINGFACE
EMBEDDING_MODEL_ID_PRIMARY=sentence-transformers/all-MiniLM-L6-v2

# Fallback LLM and Embedding Providers Configuration
GENERATION_BACKEND_FALLBACK=GEMINI
GENERATION_MODEL_ID_FALLBACK=gemini-2.5-flash
EMBEDDING_BACKEND_FALLBACK=GEMINI
EMBEDDING_MODEL_ID_FALLBACK=gemini-embedding-2

# Local Paths
HF_MODEL_DIR=~/huggingface_models
lang_detection_model=path/to/language_detector.pkl

# Qdrant Vector Database Configuration
VECTORDB_BACKEND=QDRANT
QDRANT_PATH=path/to/qdrant_data
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_IN_MEMORY=

# LangChain/LangSmith Tracing Configuration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT="Mental Rag app"

LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT="Mental Rag app"
```

## Running the Application

To start the FastAPI server locally:
```bash
python -m uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
```

Once running, you can access:
- Interactive API Documentation (Swagger UI): `http://127.0.0.1:8000/docs`
- Root API endpoint: `http://127.0.0.1:8000/`

## Running Evaluations

To run evaluation benchmarks on the RAG pipeline using the datasets in `eval/dataset/`:
```bash
python eval/eval.py
```
This runs the synthetic dataset through the pipeline and prints faithfulness, answer relevancy, and context precision scores calculated via Ragas metrics.

## API Usage

### Chat Query
Sends a message to the chatbot. Pipelines are managed per session_id to maintain conversational history.

- **URL**: `/api/v1/chat/query`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "user_message": "I have been feeling very anxious recently, what should I do?",
    "session_id": "session_user_992"
  }
  ```
- **Response**:
  ```json
  {
    "response": "Based on retrieved context..."
  }
  ```
