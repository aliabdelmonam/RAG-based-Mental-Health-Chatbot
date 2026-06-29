from fastapi import APIRouter
from src.rag import FullPipeline 
from src.stores import LLMProviderFactory
from src.core import get_settings
from src.db import VectorDBFactory
from pydantic import BaseModel
from src.rag.Rag_module.base_pipeline import ProviderBundle,BundleManager
from src.rag.Language_Detection_module import LanguageDetector


router = APIRouter()
settings = get_settings()

# Define the JSON structure your endpoint expects
class ChatQueryRequest(BaseModel):
    user_message: str
    session_id: str 

llm_provider = LLMProviderFactory(settings)
vector_db_provider = VectorDBFactory(settings)

generation_client_primary = llm_provider.create(provider=settings.GENERATION_BACKEND_PRIMAY)
generation_client_primary.set_generation_model(settings.GENERATION_MODEL_ID_PRIMARY)

embedding_client_primary = llm_provider.create(provider=settings.EMBEDDING_BACKEND_PRIMARY)
embedding_client_primary.set_embedding_model(settings.EMBEDDING_MODEL_ID_PRIMARY)

generation_client_fallback = llm_provider.create(provider=settings.GENERATION_BACKEND_FALLBACK)
generation_client_fallback.set_generation_model(settings.GENERATION_MODEL_ID_FALLBACK)

embedding_client_fallback = llm_provider.create(provider=settings.EMBEDDING_BACKEND_FALLBACK)
embedding_client_fallback.set_embedding_model(settings.EMBEDDING_MODEL_ID_FALLBACK)

qdrant_client = vector_db_provider.create(provider=settings.VECTORDB_BACKEND, embedding=embedding_client_primary)

primary = ProviderBundle(generation_client=generation_client_primary,
                          embedding_client=embedding_client_primary,
                         vector_db=qdrant_client,
                         name="Primary")
fallback = ProviderBundle(generation_client=generation_client_fallback,
                           embedding_client=embedding_client_fallback,
                           vector_db=qdrant_client,
                           name="Fallback")

client = BundleManager(primary=primary, fallback=fallback, max_retries=3)

# Instantiate LanguageDetector once globally
lang_detector = LanguageDetector(model_path=settings.lang_detection_model, threshold=0.60)

pipelines: dict[str, FullPipeline] = {}


def get_pipeline(session_id: str) -> FullPipeline:
    if session_id not in pipelines:
        pipelines[session_id] = FullPipeline(
            client=client,
            collection_name="Normal_chunking",
            top_k=1,
            lang_detector=lang_detector,
        )
    return pipelines[session_id]


@router.post("/query")
async def ask_chatbot(request_body: ChatQueryRequest):
    """
    Endpoint to interact with the RAG-based mental health chatbot.
    """
    pipeline = get_pipeline(request_body.session_id)
    result = pipeline.run(query=request_body.user_message)

    return {"response": result}