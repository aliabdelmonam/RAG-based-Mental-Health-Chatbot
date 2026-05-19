from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional



# ─────────────────────────────────────────────────────────────
# GENERATION INTERFACE
# ─────────────────────────────────────────────────────────────

class LLMGenerationInterface(ABC):
    """
    Abstract interface for any text-generation provider (Groq, OpenAI, …).

    Contract:
    - Model ID is set at construction time via set_generation_model().
    - generate_text() must always return a GenerationResponse.
    - health_check() must not raise; return False on failure.
    """

    @abstractmethod
    def set_generation_model(self, model_id: str) -> None:
        """
        Configure which model to use for generation.
        Call once after construction before any generate_text() calls.

        Args:
            model_id: Provider-specific model identifier.
                      e.g. "llama-3.3-70b-versatile" for Groq.
        """
        pass

    @abstractmethod
    def get_generation_model(self) -> str:
        """Return the currently configured generation model ID."""
        pass

    @abstractmethod
    def generate_text(
        self,
        messages:      list[Message],
        system_prompt: Optional[str]      = None,
        config:        Optional[GenerationConfig] = None,
    ) -> GenerationResponse:
        """
        Generate a response from the model.

        Args:
            messages:      Conversation history as a list of Message objects.
                           Must contain at least one user message.
            system_prompt: Optional system-level instruction injected before
                           the conversation. Keeps system logic out of messages.
            config:        Generation parameters (temperature, max_tokens, stop).
                           Falls back to GenerationConfig defaults if None.

        Returns:
            GenerationResponse with content and token usage metadata.
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Verify the provider is reachable and the model is available.
        Must never raise — return False on any failure.

        Use this at app startup and in monitoring endpoints.
        """
        pass

