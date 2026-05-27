from typing import Optional
import cohere
from cohere import Client as CohereClient
try:
    # Cohere SDK v7+: no CohereError export; base errors are subclasses of Exception
    from cohere.errors import CohereError  # type: ignore
except ImportError:  # pragma: no cover
    CohereError = Exception


from stores.LLMInterface import LLMInterface
from stores.LLMEnums import CoHereEnums
from stores.schema import (
    Message,
    GenerationConfig,
    GenerationResponse,
    EmbeddingResponse,
)
from core.logger import get_logger

logger = get_logger(__name__)


class CohereLLMProvider(LLMInterface):
    """
    Cohere provider that implements both text generation and embeddings.

    Cohere exposes both capabilities under the same SDK (v2 client),
    so it naturally fits the combined LLMInterface.

    Generation  → Cohere Chat API  (command-r-plus, command-r, …)
    Embeddings  → Cohere Embed API (embed-english-v3.0, embed-multilingual-v3.0, …)

    Usage
    -----
    provider = CohereLLMProvider(api_key="...")
    provider.set_generation_model("command-r-plus")
    provider.set_embedding_model("embed-english-v3.0")

    response  = provider.generate_text(messages=[...], system_prompt="...")
    embedding = provider.embed_query("I feel anxious lately")
    """

    # Embed input types required by Cohere's v3 embed models
    _QUERY_INPUT_TYPE = "search_query"
    _DOCUMENT_INPUT_TYPE = "search_document"

    # ─────────────────────────────────────────────────────────────
    # Construction
    # ─────────────────────────────────────────────────────────────

    def __init__(
        self,
        api_key: str,
        default_generation_model: str = "command-r-plus",
        default_embedding_model: str = "embed-english-v3.0",
    ) -> None:
        """
        Args:
            api_key:                   Your Cohere API key.
            default_generation_model:  Chat model used when set_generation_model()
                                       is never called explicitly.
            default_embedding_model:   Embed model used when set_embedding_model()
                                       is never called explicitly.
        """
        self.client: CohereClient = cohere.Client(api_key=api_key)
        self._generation_model: str = default_generation_model
        self._embedding_model: str = default_embedding_model

        logger.info("CohereLLMProvider initialized.")


    # ─────────────────────────────────────────────────────────────
    # Generation Interface
    # ─────────────────────────────────────────────────────────────

    def set_generation_model(self, model_id: str) -> None:
        """Switch the active Cohere chat model at runtime."""
        logger.debug(
            "Generation model changed: %s → %s",
            self._generation_model,
            model_id,
        )
        self._generation_model = model_id

    def get_generation_model(self) -> str:
        """Return the currently active Cohere chat model ID."""
        return self._generation_model

    def generate_text(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResponse:
        """
        Send a chat request to the Cohere Chat API.

        Cohere's Chat API separates the conversation history from the
        latest user message — the last message in `messages` is used as
        the current user turn; everything before it becomes `chat_history`.

        Args:
            messages:      Full conversation history as Message objects.
                           The last message MUST have role "user".
            system_prompt: Optional preamble injected as a system instruction.
            config:        Temperature, max_tokens, stop sequences.
                           Falls back to GenerationConfig defaults if None.

        Returns:
            GenerationResponse with generated text and token-usage metadata.

        Raises:
            ValueError:   If messages is empty or the last message is not a user turn.
            RuntimeError: Wraps any CohereError for clean upstream handling.
        """
        if not messages:
            raise ValueError("generate_text: 'messages' must not be empty.")

        if config is None:
            config = GenerationConfig()

        # Cohere distinguishes the current user message from prior history
        current_message = messages[-1]
        if current_message.role != "user":
            raise ValueError(
                "generate_text: the last message must have role 'user', "
                f"got '{current_message.role}'."
            )

        # Build chat_history using CoHereEnums for correct role values
        _role_map = {
            "system":    CoHereEnums.SYSTEM.value,
            "user":      CoHereEnums.USER.value,
            "assistant": CoHereEnums.ASSISTANT.value,   # maps to "CHATBOT"
        }
        chat_history = [
            {"role": _role_map.get(m.role, m.role.upper()), "message": m.content}
            for m in messages[:-1]
        ]

        try:
            logger.debug(
                "generate_text | model=%s | history=%d | temp=%.2f | max_tokens=%d",
                self._generation_model,
                len(chat_history),
                config.temperature,
                config.max_tokens,
            )

            response = self.client.chat(
                model=self._generation_model,
                message=current_message.content,
                chat_history=chat_history if chat_history else None,
                preamble=system_prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                stop_sequences=config.stop if config.stop else None,
            )

            usage = response.meta.tokens if response.meta else None
            input_tokens = usage.input_tokens if usage else 0
            output_tokens = usage.output_tokens if usage else 0
            finish_reason = response.finish_reason or "stop"

            logger.info(
                "generate_text OK | model=%s | in=%d out=%d tokens | finish=%s",
                self._generation_model,
                input_tokens,
                output_tokens,
                finish_reason,
            )

            return GenerationResponse(
                content=response.text or "",
                model_id=self._generation_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
            )

        except CohereError as exc:
            logger.error("generate_text FAILED | %s", exc)
            raise RuntimeError(f"Cohere generation error: {exc}") from exc

    # ─────────────────────────────────────────────────────────────
    # Embedding Interface
    # ─────────────────────────────────────────────────────────────

    def set_embedding_model(self, model_id: str) -> None:
        """Switch the active Cohere embedding model at runtime."""
        logger.debug(
            "Embedding model changed: %s → %s",
            self._embedding_model,
            model_id,
        )
        self._embedding_model = model_id

    def get_embedding_model(self) -> str:
        """Return the currently active Cohere embedding model ID."""
        return self._embedding_model

    def get_embedding_dimension(self) -> int:
        """
        Return the vector dimension for the current embedding model.

        Covers all common Cohere embed v3 models.
        Falls back to a live API probe for unknown models.
        """
        known_dims = {
            "embed-english-v3.0": 1024,
            "embed-multilingual-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-light-v3.0": 384,
            # Legacy models
            "embed-english-v2.0": 4096,
            "embed-english-light-v2.0": 1024,
            "embed-multilingual-v2.0": 768,
        }
        if self._embedding_model in known_dims:
            return known_dims[self._embedding_model]

        logger.warning(
            "get_embedding_dimension: unknown model '%s', probing via API.",
            self._embedding_model,
        )
        probe = self.embed_query("probe")
        return probe.dimension

    def embed_query(self, text: str) -> EmbeddingResponse:
        """
        Embed a single user query (inference time).

        Uses Cohere's `search_query` input_type for best retrieval quality
        with v3 embedding models.

        Args:
            text: The user's raw input string.

        Returns:
            EmbeddingResponse with a single embedding vector.
        """
        return self._embed([text], input_type=self._QUERY_INPUT_TYPE)

    def embed_documents(self, texts: list[str]) -> EmbeddingResponse:
        """
        Embed a batch of document chunks (indexing time).

        Uses Cohere's `search_document` input_type for best retrieval quality
        with v3 embedding models.

        Args:
            texts: List of document chunks to embed.

        Returns:
            EmbeddingResponse with one vector per input text.
        """
        if not texts:
            raise ValueError("embed_documents: 'texts' must not be empty.")
        return self._embed(texts, input_type=self._DOCUMENT_INPUT_TYPE)

    # ─────────────────────────────────────────────────────────────
    # Health Check
    # ─────────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """
        Verify the Cohere API is reachable and both models are accessible.

        Returns:
            True  — generation and embedding APIs responded successfully.
            False — any connectivity, authentication, or model error.
        """
        try:
            # Minimal generation ping
            self.client.chat(
                model=self._generation_model,
                message="ping",
                max_tokens=1,
            )
            # Minimal embedding ping
            self.client.embed(
                model=self._embedding_model,
                texts=["ping"],
                input_type=self._QUERY_INPUT_TYPE,
            )
            logger.info(
                "health_check OK | gen=%s emb=%s",
                self._generation_model,
                self._embedding_model,
            )
            return True
        except CohereError as exc:
            logger.warning("health_check FAILED | %s", exc)
            return False

    # ─────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────

    def _embed(self, texts: list[str], input_type: str) -> EmbeddingResponse:
        """
        Internal helper that calls the Cohere Embed API.

        Args:
            texts:      One or more strings to embed.
            input_type: "search_query" or "search_document" — required by
                        Cohere v3 models to apply the correct prompt prefix.

        Returns:
            EmbeddingResponse.

        Raises:
            RuntimeError: wraps any CohereError.
        """
        try:
            logger.debug(
                "_embed | model=%s | input_type=%s | n_texts=%d",
                self._embedding_model,
                input_type,
                len(texts),
            )

            response = self.client.embed(
                model=self._embedding_model,
                texts=texts,
                input_type=input_type,
            )

            vectors: list[list[float]] = response.embeddings
            dim = len(vectors[0]) if vectors else 0

            logger.info(
                "_embed OK | model=%s | n=%d | dim=%d",
                self._embedding_model,
                len(vectors),
                dim,
            )

            return EmbeddingResponse(
                embeddings=vectors,
                model_id=self._embedding_model,
                dimension=dim,
            )

        except CohereError as exc:
            logger.error("_embed FAILED | %s", exc)
            raise RuntimeError(f"Cohere embedding error: {exc}") from exc
