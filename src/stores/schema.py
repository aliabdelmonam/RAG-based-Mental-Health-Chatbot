from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────
# DATA CLASSES  —  clean input/output contracts
# ─────────────────────────────────────────────────────────────

@dataclass
class Message:
    """Represents a single turn in a conversation."""
    role: str          # "system" | "user" | "assistant"
    content: str


@dataclass
class GenerationConfig:
    """
    Runtime parameters for a generation call.
    Defaults are tuned for a mental health chatbot:
    - low temperature  → calm, consistent, predictable responses
    - moderate tokens  → complete answers without rambling
    """
    temperature: float       = 0.3
    max_output_tokens :  int         = 1024
    stop:        list[str]   = field(default_factory=list)


@dataclass
class GenerationResponse:
    """Structured output returned by every generate_text() call."""
    content:      str
    model_id:     str
    input_tokens:  int
    output_tokens: int
    finish_reason: str   # "stop" | "length" | "error"



@dataclass
class EmbeddingResponse:
    """Structured output returned by every embed_*() call."""
    embeddings: list[list[float]]   # shape: (n_texts, embedding_dim)
    model_id:   str
    dimension:  int

