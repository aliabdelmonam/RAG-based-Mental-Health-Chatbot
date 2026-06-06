from fastapi import APIRouter
from src.rag import FullPipeline 
from src.stores import LLMProviderFactory
from src.core import get_settings
from src.db import VectorDBFactory
from pydantic import BaseModel


router = APIRouter()
settings = get_settings()

# Define the JSON structure your endpoint expects
class ChatQueryRequest(BaseModel):
    user_message: str

# Initialize provider factories (Safe to do globally as they don't open network sockets)
llm_provider = LLMProviderFactory(settings)
vector_db_provider = VectorDBFactory(settings)

# Create generation + embedding clients (Safe to instantiate globally)
generation_client = llm_provider.create(provider=settings.GENERATION_BACKEND)
embedding_client = llm_provider.create(provider=settings.EMBEDDING_BACKEND)

# Configure models (Safe to do globally)
generation_client.set_generation_model(settings.GENERATION_MODEL_ID)
embedding_client.set_embedding_model(settings.EMBEDDING_MODEL_ID)

# Create the vector DB client reference globally, but DO NOT call .connect() here!
qdrant_client = vector_db_provider.create(provider=settings.VECTORDB_BACKEND, embedding=embedding_client)

pipeline = FullPipeline(
        generation_client=generation_client,
        embedding_client=embedding_client,
        vector_db_client=qdrant_client, # Uses the client connected via main.py lifespan
        collection_name="Normal_chunking",
        top_k=1,
    )
@router.post("/query")
async def ask_chatbot(request_body: ChatQueryRequest):
    """
    Endpoint to interact with the RAG-based mental health chatbot.
    """
    result = pipeline.run(query=request_body.user_message)

    return {"response": result}