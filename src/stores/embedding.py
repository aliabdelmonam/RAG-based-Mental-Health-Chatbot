from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.stores.schema import EmbeddingResponse


# ─────────────────────────────────────────────────────────────
# EMBEDDING INTERFACE
# ─────────────────────────────────────────────────────────────

class LLMEmbeddingInterface(ABC):
    """
    Abstract interface for any embedding provider (SentenceTransformers, …).

    embed_query()     — optimised for a single user query at inference time.
    embed_documents() — optimised for a batch of documents at indexing time.

    These are intentionally separate because some models (e.g. E5, BGE)
    use different prompt prefixes for queries vs documents to improve
    retrieval quality.
    """

    @abstractmethod
    def set_embedding_model(self, model_id: str) -> None:
        """
        Configure which embedding model to use.

        Args:
            model_id: Model name or path.
                      e.g. "sentence-transformers/all-MiniLM-L6-v2"
        """
        pass

    @abstractmethod
    def get_embedding_model(self) -> str:
        """Return the currently configured embedding model ID."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> EmbeddingResponse:
        """
        Embed a single user query at inference time.

        Args:
            text: The user's input message.

        Returns:
            EmbeddingResponse containing a single embedding vector.
        """
        pass

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> EmbeddingResponse:
        """
        Embed a batch of documents at indexing time.

        Args:
            texts: List of document chunks to embed.
                   All texts are embedded in a single batched call
                   for efficiency.

        Returns:
            EmbeddingResponse with one embedding vector per input text.
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Return the vector dimension produced by the current model.
        Qdrant requires this when creating a collection.
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Verify the model is loaded and functional.
        Must never raise — return False on any failure.
        """
        pass
