from typing import Optional
from groq import Groq, GroqError

from stores.generation import LLMGenerationInterface
from stores.LLMEnums import GroqEnums
from stores.schema import (
    Message,
    GenerationConfig,
    GenerationResponse,
)
from core.logger import get_logger

logger = get_logger(f'GroqProvider:')


class GroqLLMProvider(LLMGenerationInterface):
    """
    Groq provider for ultra-fast text generation via the Groq Cloud API.

    Groq does NOT expose an embeddings endpoint, so this class implements
    only LLMGenerationInterface (not the combined LLMInterface).
    For embeddings use a dedicated provider (e.g. SentenceTransformers).

    Usage
    -----
    provider = GroqLLMProvider(api_key="gsk_...")
    provider.set_generation_model("llama-3.3-70b-versatile")

    response = provider.generate_text(
        messages=[Message(role="user", content="How are you feeling today?")],
        system_prompt="You are a compassionate mental health assistant.",
    )
    print(response.content)
    """

    # Default model — fast and high-quality for conversational tasks
    DEFAULT_MODEL = "openai/gpt-oss-safeguard-20b"

    # ─────────────────────────────────────────────────────────────
    # Construction
    # ─────────────────────────────────────────────────────────────

    def __init__(
        self,
        api_key: str,
        generation_model: str,
    ) -> None:
        """
        Args:
            api_key:                  Your Groq API key (gsk_…).
            generation_model: Model used when set_generation_model()
                                      is never called explicitly.
                                      Defaults to llama-3.3-70b-versatile.
        """
        self.client: Groq = Groq(api_key=api_key)
        self._generation_model: str = generation_model

        logger.info("GroqLLMProvider initialized.")


    # ─────────────────────────────────────────────────────────────
    # Generation Interface
    # ─────────────────────────────────────────────────────────────

    def set_generation_model(self, model_id: str) -> None:
        """Switch the active Groq model at runtime."""
        logger.debug(
            "Generation model changed: %s → %s",
            self._generation_model,
            model_id,
        )
        self._generation_model = model_id

    def get_generation_model(self) -> str:
        """Return the currently active Groq model ID."""
        return self._generation_model

    def generate_text(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResponse:
        """
        Send a chat-completion request to the Groq API.

        Args:
            messages:      Conversation history (user / assistant turns).
                           Must contain at least one user message.
            system_prompt: Optional system-level instruction injected before
                           the conversation history.
            config:        Generation parameters (temperature, max_tokens, stop).
                           Falls back to GenerationConfig defaults if None.

        Returns:
            GenerationResponse with generated text and token-usage metadata.

        Raises:
            RuntimeError: wraps any GroqError so callers get a clean exception.
        """
        if config is None:
            config = GenerationConfig()

        # Build the API message list
        api_messages: list[dict] = []

        if system_prompt:
            api_messages.append({"role": GroqEnums.SYSTEM.value, "content": system_prompt})

        # Map internal role names to Groq role values via GroqEnums
        _role_map = {
            "system":    GroqEnums.SYSTEM.value,
            "user":      GroqEnums.USER.value,
            "assistant": GroqEnums.ASSISTANT.value,
        }
        for msg in messages:
            api_messages.append({"role": _role_map.get(msg.role, msg.role), "content": msg.content})

        try:
            logger.debug(
                "generate_text | model=%s | messages=%d | temp=%.2f | max_tokens=%d",
                self._generation_model,
                len(api_messages),
                config.temperature,
                config.max_tokens,
            )

            response = self.client.chat.completions.create(
                model=self._generation_model,
                messages=api_messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                stop=config.stop if config.stop else None,
            )

            choice = response.choices[0]
            usage = response.usage

            logger.info(
                "generate_text OK | model=%s | in=%d out=%d tokens | finish=%s",
                response.model,
                usage.prompt_tokens,
                usage.completion_tokens,
                choice.finish_reason,
            )

            return GenerationResponse(
                content=choice.message.content or "",
                model_id=response.model,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                finish_reason=choice.finish_reason,
            )

        except GroqError as exc:
            logger.error("generate_text FAILED | %s", exc)
            raise RuntimeError(f"Groq generation error: {exc}") from exc

    # ─────────────────────────────────────────────────────────────
    # Health Check
    # ─────────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """
        Verify the Groq API is reachable and the configured model is available.

        Sends a minimal 1-token request to avoid unnecessary costs.

        Returns:
            True  — API responded successfully.
            False — any connectivity, auth, or model error.
        """
        try:
            self.client.chat.completions.create(
                model=self._generation_model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            logger.info("health_check OK | model=%s", self._generation_model)
            return True
        except GroqError as exc:
            logger.warning("health_check FAILED | %s", exc)
            return False
