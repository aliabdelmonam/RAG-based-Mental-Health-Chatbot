from src.rag.Rag_module import RAGPipeline, NormalPipeline
from src.rag.Intent_Classifier_module import IntentClassifier, IntentLabel
from src.rag.Language_Detection_module import LanguageDetector
from src.stores import LLMProviderFactory,GenerationConfig
from src.core.Config import get_settings
settings = get_settings()
from src.db import VectorDBFactory
from typing import Optional
from src.core.logger import get_logger
from src.rag.Rag_module.conversation import ConversationHistory

logger = get_logger(f"FullPipeline:")

LANG_MODEL_PATH = r"C:\Users\aliab\Downloads\language_detector.pkl"

llm_provider = LLMProviderFactory(settings)
vector_db_provider = VectorDBFactory(settings)

generation_client = llm_provider.create(provider=settings.GENERATION_BACKEND)
generation_client.set_generation_model(settings.GENERATION_MODEL_ID)

embedding_client = llm_provider.create(provider=settings.EMBEDDING_BACKEND)
embedding_client.set_embedding_model(settings.EMBEDDING_MODEL_ID)

qdrant_client = vector_db_provider.create(provider=settings.VECTORDB_BACKEND)


class FullPipeline:
    def __init__(
        self,
        generation_client,
        embedding_client,
        vector_db_client,
        collection_name: str,
        top_k: int = 5,
    ):
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.vector_db_client = vector_db_client
        self.collection_name = collection_name
        self.top_k = top_k

        self._lang_detector = LanguageDetector(model_path=settings.lang_detection_model, threshold=0.60)
        self.intent_classifier = IntentClassifier(generation_client=generation_client, language_detector=self._lang_detector)

        self.history = ConversationHistory(max_turns=10)

    def run(self, query: str):
        # 1) Classify intent
        if self.intent_classifier:
            intent_raw = self.intent_classifier.classify(query, chat_history=self.history.get())
            logger.info(f"Intent Raw: {intent_raw}")
        else:
            intent_raw = "unknown"
        
        # 2) QueryRoute
        if intent_raw.requires_rag or intent_raw=='unknown':

            logger.info(f"Running RAG Pipeline for query: {query}")
            Config = GenerationConfig(max_new_tokens=100, temperature=0.3)
            rag_pipeline = RAGPipeline(
                generation_client=self.generation_client,
                embedding_client=self.embedding_client,
                vector_db_client=self.vector_db_client,
                intent_classifier=self.intent_classifier,
                collection_name=self.collection_name,
                top_k=self.top_k,
                generation_config=Config
            )
            emotion = 'unknown'
            return rag_pipeline.run(emotion, query, history=self.history)
        elif intent_raw.intent == IntentLabel.OUT_OF_SCOPE:
            logger.info(f"Out of scope intent detected for query: {query}. predetermined response will be returned.")
            return "I'm sorry, but this chatbot is designed only to help with mental health questions and emotional support. I won't be able to assist with home repairs."
        else:
            logger.info(f"Running Normal Generation Pipeline for query: {query}")
            Config = GenerationConfig(max_new_tokens=100, temperature=0.3)
            normal_pipeline = NormalPipeline(
                generation_client=self.generation_client,
                generation_config=Config)
            
            return normal_pipeline.run(query=query)