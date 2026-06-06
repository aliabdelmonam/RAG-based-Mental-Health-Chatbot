from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.v1.router import api_router


from src.api.v1.endpoints.chat import qdrant_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    Guarantees database connections only occur inside the active worker process.
    """
    print("🚀 [Startup] Initializing secure connections...")
    
    # 1. Connect to Qdrant safely inside the lifespan hook
    qdrant_client.connect()
    print("✅ Qdrant database connected successfully.")

    # 2. Optional: Run backend health checks on startup
    # print("🔍 Running LLM & Embedding provider health checks...")
    # embedding_client.health_check()
    # generation_client.health_check()
    
    yield  # The application is now running and ready to receive API requests
    
    print("🛑 [Shutdown] Cleaning up application resources...")


# Initialize the FastAPI app with the lifespan context
app = FastAPI(
    title="RAG-Based Mental Health Chatbot API",
    version="1.0.0",
    lifespan=lifespan
)

# Include API endpoints under version 1 route group
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
def root():
    return {
        "message": "Welcome to the RAG Mental Health Chatbot API. Head to /docs for Swagger UI."
    }