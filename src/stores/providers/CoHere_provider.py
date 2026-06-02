from typing import Optional
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

from src.stores.LLMInterface import LLMInterface
from src.stores.LLMEnums import CoHereEnums
from src.stores.schema import (
    Message,
    GenerationConfig,
    GenerationResponse,
    EmbeddingResponse,
)
from src.core.logger import get_logger

logger = get_logger(__name__)


class CohereLLMProvider(LLMInterface):
    """
    Cohere provider that implements both text generation and embeddings
    via LangChain + Cohere.

    Generation  → ChatCohere        (command-r-plus, command-r, …)
    Embeddings  → CohereEmbeddings  (embed-english-v3.0, embed-multilingual-v3.0, …)

    Usage
    -----
    provider = CohereLLMProvider(api_key="...")
    provider.set_generation_model("command-r-plus")
    provider.set_embedding_model("embed-english-v3.0")

    response  = provider.generate_text(messages=[...], system_prompt="...")
    embedding = provider.embed_query("I feel anxious lately")
    """

    _QUERY_INPUT_TYPE    = "search_query"
    _DOCUMENT_INPUT_TYPE = "search_document"

    # ─────────────────────────────────────────────────────────────
    # Construction
    # ─────────────────────────────────────────────────────────────

    def __init__(
        self,
        api_key: str,
        generation_model: str = "command-r-plus",
        embedding_model: str = "embed-english-v3.0",
    ) -> None:
        self._api_key = api_key
        self._generation_model = generation_model
        self._embedding_model  = embedding_model
        # self._build_generation_client()
        # self._build_embedding_client()
        logger.info("CohereLLMProvider initialized.")

    def _build_generation_client(self) -> None:
        """Instantiate (or re-instantiate) the LangChain ChatCohere client."""
        self.generation_client = ChatCohere(
            cohere_api_key=self._api_key,
            model=self._generation_model,
        )

    def _build_embedding_client(self) -> None:
        """Instantiate (or re-instantiate) the LangChain CohereEmbeddings client."""
        self.embedding_client = CohereEmbeddings(
            cohere_api_key=self._api_key,
            model=self._embedding_model,
        )

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
        self._build_generation_client()  # rebuild with new model

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
        Send a chat request via LangChain + Cohere.

        Args:
            messages:      Full conversation history as Message objects.
                           The last message MUST have role "user".
            system_prompt: Optional preamble injected as a system instruction.
            config:        Temperature, max_tokens, stop sequences.
                           Falls back to GenerationConfig defaults if None.

        Returns:
            GenerationResponse with generated text and token-usage metadata.

        Raises:
            ValueError:   If messages is empty or last message is not a user turn.
            RuntimeError: Wraps any exception for clean upstream handling.
        """
        if not messages:
            raise ValueError("generate_text: 'messages' must not be empty.")

        if config is None:
            config = GenerationConfig()

        if messages[-1].role != "user":
            raise ValueError(
                "generate_text: the last message must have role 'user', "
                f"got '{messages[-1].role}'."
            )

        # ── Map internal Message objects → LangChain message types ──
        _role_map = {
            "system":    SystemMessage,
            "user":      HumanMessage,
            "assistant": AIMessage,
        }

        lc_messages: list[BaseMessage] = []

        if system_prompt:
            lc_messages.append(SystemMessage(content=system_prompt))

        for msg in messages:
            msg_class = _role_map.get(msg.role, HumanMessage)
            lc_messages.append(msg_class(content=msg.content))

        try:
            logger.debug(
                "generate_text | model=%s | messages=%d | temp=%.2f | max_tokens=%d",
                self._generation_model,
                len(lc_messages),
                config.temperature,
                config.max_tokens,
            )

            response = self.generation_client.invoke(
                lc_messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                **({"stop_sequences": config.stop} if config.stop else {}),
            )

            # ── Extract usage metadata ──────────────────────────────
            usage         = response.response_metadata.get("token_count", {})
            input_tokens  = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("response_tokens", 0)
            finish_reason = response.response_metadata.get("finish_reason", "stop")

            logger.info(
                "generate_text OK | model=%s | in=%d out=%d tokens | finish=%s",
                self._generation_model,
                input_tokens,
                output_tokens,
                finish_reason,
            )

            return GenerationResponse(
                content=response.content or "",
                model_id=self._generation_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
            )

        except Exception as exc:
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
        self._build_embedding_client()  # rebuild with new model

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
            "embed-english-v3.0":              1024,
            "embed-multilingual-v3.0":         1024,
            "embed-english-light-v3.0":         384,
            "embed-multilingual-light-v3.0":    384,
            # Legacy models
            "embed-english-v2.0":              4096,
            "embed-english-light-v2.0":        1024,
            "embed-multilingual-v2.0":          768,
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

        Args:
            text: The user's raw input string.

        Returns:
            EmbeddingResponse with a single embedding vector.
        """
        return self._embed([text], input_type=self._QUERY_INPUT_TYPE)

    def embed_documents(self, texts: list[str]) -> EmbeddingResponse:
        """
        Embed a batch of document chunks (indexing time).

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
        generation_ok = False
        embedding_ok = False

        try:
            self.generation_client.invoke(
                [HumanMessage(content="ping")],
                max_tokens=1,
            )
            generation_ok = True
            logger.info("Generation health check OK | model=%s", self._generation_model)

        except Exception as exc:
            logger.warning("Generation health check FAILED | %s", exc)

        try:
            self.embedding_client.embed_query("ping")
            embedding_ok = True
            logger.info("Embedding health check OK | model=%s", self._embedding_model)

        except Exception as exc:
            logger.warning("Embedding health check FAILED | %s", exc)

        return generation_ok and embedding_ok

    # ─────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────

    def _embed(self, texts: list[str], input_type: str) -> EmbeddingResponse:
        """
        Internal helper that calls the Cohere Embed API via LangChain.

        Args:
            texts:      One or more strings to embed.
            input_type: "search_query" or "search_document".

        Returns:
            EmbeddingResponse.

        Raises:
            RuntimeError: wraps any exception.
        """
        try:
            logger.debug(
                "_embed | model=%s | input_type=%s | n_texts=%d",
                self._embedding_model,
                input_type,
                len(texts),
            )

            # Rebuild client with the correct input_type for this call
            typed_client = CohereEmbeddings(
                cohere_api_key=self._api_key,
                model=self._embedding_model,
                input_type=input_type,
            )

            if len(texts) == 1:
                vectors = [typed_client.embed_query(texts[0])]
            else:
                vectors = typed_client.embed_documents(texts)

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

        except Exception as exc:
            logger.error("_embed FAILED | %s", exc)
            raise RuntimeError(f"Cohere embedding error: {exc}") from exc