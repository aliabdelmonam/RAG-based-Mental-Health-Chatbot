from __future__ import annotations

from typing import  TYPE_CHECKING
from src.stores import GenerationConfig
from src.core.Config import get_settings
from src.core.logger import get_logger
from src.rag.Rag_module.conversation import ConversationHistory

if TYPE_CHECKING:
    from src.rag.Rag_module.base_pipeline import BundleManager

from src.rag.Rag_module import RAGPipeline, NormalPipeline
from src.rag.Language_Detection_module import LanguageDetector

settings = get_settings()
logger = get_logger(f"FullPipeline:")

LANG_MODEL_PATH = r"C:\Users\aliab\Downloads\language_detector.pkl"


class FullPipeline:
    def __init__(
        self,
        client: BundleManager,
        collection_name: str,
        top_k: int = 5,
    ):
        # Import at runtime to avoid circular dependency
        from src.rag.Intent_Classifier_module import IntentClassifier
        
        self.client=client
        self.collection_name = collection_name
        self.top_k = top_k

        self._lang_detector = LanguageDetector(model_path=settings.lang_detection_model, threshold=0.60)
        self.intent_classifier = IntentClassifier(client=client, language_detector=self._lang_detector)

        self.history = ConversationHistory(max_turns=10)

    def run(self, query: str):
        # Import at runtime to avoid circular dependency
        from src.rag.Intent_Classifier_module import IntentLabel
        
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
                client = self.client,
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
                client=self.client,
                generation_config=Config)
            
            return normal_pipeline.run(query=query)