from src.stores.LLMInterface import LLMInterface
from typing import Optional
from src.core.logger import get_logger
import time
from dataclasses import dataclass
from src.db.vector_db_interface import VectorDBInterface
from src.db.chunking.retrieve import Retrieve
logger = get_logger(f"BasePipeline:")

@dataclass
class ProviderBundle:
    name:str
    generation_client: LLMInterface
    embedding_client: LLMInterface
    vector_db: Optional[VectorDBInterface] = None

class BundleManager:
    """
    Single entry point for all provider operations.
    Handles retry and fallback transparently.
    Pipelines only ever talk to this class.
    """
    def __init__(
        self,
        primary: ProviderBundle,
        fallback: Optional[ProviderBundle] = None,
        max_retries: int = 3,
    ):
        self._primary = primary
        self._fallback = fallback
        self.max_retries = max_retries

    def generate(self, messages, config, enable_tools=False):
        return self._run(
            lambda b: b.generation_client.generate_text(messages, config, enable_tools),
            operation="generate"
        )

    def embed(self,query):
        return self._run(
            lambda b:b.embedding_client.embed_query(query),
            operation='embed'
        )
    
    def embed_and_search(self, query, collection_name, top_k, top_q=1):
        return self._run(
            lambda b: Retrieve(b.vector_db).search_with(
                query_vector=b.embedding_client.embed_query(query),
                collection_name=collection_name,
                top_k=top_k,
                top_q=top_q,
            ),
            operation="embed_and_search"
        )
    
    def search(self, query_vector, collection_name, top_k, top_q=1):
        return self._run(
            lambda b: Retrieve(b.vector_db).search_with(
                query_vector=query_vector,
                collection_name=collection_name,
                top_k=top_k,
                top_q=top_q,
            ),
            operation="embed_and_search"
        )

    def _run(self, fn, operation: str):
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                return fn(self._primary)
            except Exception as e:
                last_exc = e
                logger.warning(f"{operation} primary attempt {attempt + 1} failed: {e}")
                time.sleep(0.5 * (attempt + 1))

        if self._fallback:
            logger.warning(f"{operation} switching to fallback.")
            try:
                return fn(self._fallback)
            except Exception as e:
                raise RuntimeError(f"Both primary and fallback failed for {operation}.") from e

        raise RuntimeError(f"{operation} primary failed after {self.max_retries} attempts.") from last_exc
    
    def _safe_check(self, fn) -> bool:
        try:
            return fn()
        except Exception:
            return False
        
    def __getattr__(self, name):
        for client in [self._primary.generation_client, self._primary.embedding_client, self._primary.vector_db]:
            if hasattr(client, name):
                return getattr(client, name)
        raise AttributeError(f"BundleManager has no attribute '{name}'")
    
    def _bundle_health(self, bundle: ProviderBundle) -> dict:
        return {
            "generation": self._safe_check(bundle.generation_client.health_check),
            "embedding":  self._safe_check(bundle.embedding_client.health_check),
            "vector_db":  self._safe_check(bundle.vector_db.health_check),
        }
    
    def health_check(self) -> dict:
        return {
            "primary":  self._bundle_health(self._primary),
            "fallback": self._bundle_health(self._fallback) if self._fallback else None,
        }
    
   