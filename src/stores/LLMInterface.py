from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from stores.generation import LLMGenerationInterface
from stores.embedding import LLMEmbeddingInterface



# ─────────────────────────────────────────────────────────────
# COMBINED INTERFACE  (optional — use only if one provider
# genuinely supports both generation AND embeddings)
# ─────────────────────────────────────────────────────────────

class LLMInterface(LLMGenerationInterface, LLMEmbeddingInterface):
    """
    Combined interface for providers that support both generation
    and embeddings under a single client.

    In your current stack this will likely NOT be used:
    - Generation  → Groq         (implements LLMGenerationInterface)
    - Embeddings  → SentenceTransformers (implements LLMEmbeddingInterface)

    Keep this here for future flexibility — e.g. if you switch to
    OpenAI which supports both via one SDK.
    """
    pass