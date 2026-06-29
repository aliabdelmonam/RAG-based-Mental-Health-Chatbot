from typing import Optional
from openai import OpenAI, OpenAIError

from src.stores.LLMInterface import LLMInterface
from src.stores.LLMEnums import OpenAIEnums
from src.stores.schema import (
    Message,
    GenerationConfig,
    GenerationResponse,
    EmbeddingResponse,
)
from src.core.logger import get_logger

logger = get_logger(__name__)


class OpenAILLMProvider(LLMInterface):
    """
    OpenAI provider that implements both text generation and embeddings.

    OpenAI is one of the few providers that exposes both capabilities
    under the same SDK, so it naturally fits the combined LLMInterface.

    Usage
    -----
    provider = OpenAILLMProvider(api_key="sk-...")
    provider.set_generation_model("gpt-4o-mini")
    provider.set_embedding_model("text-embedding-3-small")

    response = provider.generate_text(messages=[...])
    embedding = provider.embed_query("how are you feeling today?")
    """

    # ─────────────────────────────────────────────────────────────
    # Construction
    # ─────────────────────────────────────────────────────────────

    def __init__(self, api_key: str, default_generation_model: str = "gpt-4o-mini",
                 default_embedding_model: str = "text-embedding-3-small") -> None:
        """
        Args:
            api_key:                   Your OpenAI secret key (sk-…).
            default_generation_model:  Chat model used when set_generation_model()
                                       is never called explicitly.
            default_embedding_model:   Embedding model used when set_embedding_model()
                                       is never called explicitly.
        """
        self.client: OpenAI = OpenAI(api_key=api_key)
        self._generation_model: str = default_generation_model
        self._embedding_model: str = default_embedding_model

        logger.info("OpenAILLMProvider initialized.")


    # ─────────────────────────────────────────────────────────────
    # Generation Interface
    # ─────────────────────────────────────────────────────────────

    def set_generation_model(self, model_id: str) -> None:
        """Switch the chat-completion model at runtime."""
        logger.debug("Generation model changed: %s -> %s", self._generation_model, model_id)
        self._generation_model = model_id

    def get_generation_model(self) -> str:
        """Return the currently active chat-completion model ID."""
        return self._generation_model

    def generate_text(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResponse:
        """
        Send a chat-completion request to OpenAI.

        Args:
            messages:      Conversation history (user / assistant turns).
            system_prompt: If provided, prepended as a system message.
            config:        Temperature, max_tokens, stop sequences.
                           Defaults to GenerationConfig() if None.

        Returns:
            GenerationResponse with generated text + token-usage metadata.

        Raises:
            RuntimeError: wraps any OpenAIError so callers get a clean exception.
        """
        if config is None:
            config = GenerationConfig()

        # Build the messages list for the API
        api_messages: list[dict] = []

        if system_prompt:
            api_messages.append({"role": OpenAIEnums.SYSTEM.value, "content": system_prompt})

        # Map internal role names to OpenAI role values via OpenAIEnums
        _role_map = {
            "system":    OpenAIEnums.SYSTEM.value,
            "user":      OpenAIEnums.USER.value,
            "assistant": OpenAIEnums.ASSISTANT.value,
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

        except OpenAIError as exc:
            logger.error("generate_text FAILED | %s", exc)
            raise RuntimeError(f"OpenAI generation error: {exc}") from exc

    # ─────────────────────────────────────────────────────────────
    # Embedding Interface
    # ─────────────────────────────────────────────────────────────

    def set_embedding_model(self, model_id: str) -> None:
        """Switch the embedding model at runtime."""
        logger.debug("Embedding model changed: %s -> %s", self._embedding_model, model_id)
        self._embedding_model = model_id

    def get_embedding_model(self) -> str:
        """Return the currently active embedding model ID."""
        return self._embedding_model

    def get_embedding_dimension(self) -> int:
        """
        Return the vector dimension for the current embedding model.

        Covers the most common OpenAI embedding models.
        Falls back to a live API call for unknown models.
        """
        known_dims = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        if self._embedding_model in known_dims:
            return known_dims[self._embedding_model]

        # For unknown / future models: embed a dummy string and measure
        logger.warning(
            "get_embedding_dimension: unknown model '%s', probing via API.",
            self._embedding_model,
        )
        probe = self.embed_query("probe")
        return probe.dimension

    def embed_query(self, text: str) -> EmbeddingResponse:
        """
        Embed a single user query (inference time).

        Args:
            text: The user's raw input string.

        Returns:
            EmbeddingResponse with a single embedding vector.
        """
        return self._embed([text])

    def embed_documents(self, texts: list[str]) -> EmbeddingResponse:
        """
        Embed a batch of document chunks (indexing time).

        Args:
            texts: List of text chunks to embed.

        Returns:
            EmbeddingResponse with one vector per input text.
        """
        if not texts:
            raise ValueError("embed_documents: 'texts' must not be empty.")
        return self._embed(texts)

    # ─────────────────────────────────────────────────────────────
    # Health Check
    # ─────────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """
        Verify the OpenAI API is reachable and the configured models exist.

        Returns:
            True  — both generation and embedding models are accessible.
            False — any connectivity or authentication failure.
        """
        try:
            # Lightweight generation ping (1 token output)
            self.client.chat.completions.create(
                model=self._generation_model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            # Lightweight embedding ping
            self.client.embeddings.create(
                model=self._embedding_model,
                input="ping",
            )
            logger.info("health_check OK | gen=%s emb=%s", self._generation_model, self._embedding_model)
            return True
        except OpenAIError as exc:
            logger.warning("health_check FAILED | %s", exc)
            return False

    # ─────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> EmbeddingResponse:
        """
        Internal helper that calls the OpenAI Embeddings API.

        Args:
            texts: One or more strings to embed.

        Returns:
            EmbeddingResponse.

        Raises:
            RuntimeError: wraps any OpenAIError.
        """
        try:
            logger.debug("_embed | model=%s | n_texts=%d", self._embedding_model, len(texts))

            response = self.client.embeddings.create(
                model=self._embedding_model,
                input=texts,
            )

            vectors = [item.embedding for item in response.data]
            dim = len(vectors[0]) if vectors else 0

            logger.info(
                "_embed OK | model=%s | n=%d | dim=%d",
                response.model,
                len(vectors),
                dim,
            )

            return EmbeddingResponse(
                embeddings=vectors,
                model_id=response.model,
                dimension=dim,
            )

        except OpenAIError as exc:
            logger.error("_embed FAILED | %s", exc)
            raise RuntimeError(f"OpenAI embedding error: {exc}") from exc