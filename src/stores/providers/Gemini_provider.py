from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

from src.stores.LLMInterface import LLMInterface
from src.stores.schema import (
    Message,
    GenerationConfig,
    GenerationResponse,
    EmbeddingResponse,
)
from src.core.logger import get_logger

logger = get_logger(__name__)


class GeminiLLMProvider(LLMInterface):
    """
    Google Gemini provider via LangChain (langchain-google-genai).

    Generation  → ChatGoogleGenerativeAI
    Embeddings  → GoogleGenerativeAIEmbeddings

    Install
    -------
    pip install langchain-google-genai

    Usage
    -----
    provider = GeminiLLMProvider(api_key="AIza...")
    provider.set_generation_model("gemini-2.0-flash")
    provider.set_embedding_model("models/text-embedding-004")

    response  = provider.generate_text(messages=[...])
    embedding = provider.embed_query("how are you feeling today?")
    """

    # ─────────────────────────────────────────────────────────────
    # Construction
    # ─────────────────────────────────────────────────────────────

    def __init__(
        self,
        api_key: str,
        # generation_model: str,
        # embedding_model: str,
    ) -> None:
        self._api_key = api_key
        self._generation_model = None
        self._embedding_model  = None
        logger.info("GeminiLLMProvider initialized (langchain-google-genai).")

    def _build_generation_client(self) -> None:
        """Instantiate (or re-instantiate) the LangChain ChatGoogleGenerativeAI client."""
        if self._generation_model is None:
            logger.warning("_build_generation_client: generation model not set, use set_generation_model() to set one.")
            return
        self.generation_client = ChatGoogleGenerativeAI(
            model=self._generation_model,
            google_api_key=self._api_key,
        )

    def _build_embedding_client(self) -> None:
        """Instantiate (or re-instantiate) the LangChain GoogleGenerativeAIEmbeddings client."""
        if self._embedding_model is None:
            logger.warning("_build_embedding_client: embedding model not set, use set_embedding_model() to set one.")
            return
        self.embedding_client = GoogleGenerativeAIEmbeddings(
            model=self._embedding_model,
            google_api_key=self._api_key,
        )

    # ─────────────────────────────────────────────────────────────
    # Generation Interface
    # ─────────────────────────────────────────────────────────────

    def set_generation_model(self, model_id: str) -> None:
        logger.debug("Generation model changed: %s → %s", self._generation_model, model_id)
        self._generation_model = model_id
        self._build_generation_client()

    def get_generation_model(self) -> str:
        return self._generation_model

    def generate_text(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResponse:
        """
        Send a generation request via LangChain + Gemini.

        Args:
            messages:      Conversation history (user / assistant turns).
            system_prompt: If provided, injected as a SystemMessage.
            config:        Temperature, max_tokens, stop sequences.
                           Defaults to GenerationConfig() if None.

        Returns:
            GenerationResponse with generated text + token-usage metadata.

        Raises:
            RuntimeError: wraps any API error so callers get a clean exception.
        """
        if config is None:
            config = GenerationConfig()

        # ── Map internal Message → LangChain message types ──────
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
                max_output_tokens=config.max_tokens,
                **({"stop_sequences": config.stop} if config.stop else {}),
            )

            # ── Extract usage metadata ──────────────────────────
            usage         = response.response_metadata.get("usage_metadata", {})
            input_tokens  = usage.get("prompt_token_count", 0)
            output_tokens = usage.get("candidates_token_count", 0)
            finish_reason = response.response_metadata.get("finish_reason", "STOP")

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
            raise RuntimeError(f"Gemini generation error: {exc}") from exc

    # ─────────────────────────────────────────────────────────────
    # Embedding Interface
    # ─────────────────────────────────────────────────────────────

    def set_embedding_model(self, model_id: str) -> None:
        logger.debug("Embedding model changed: %s → %s", self._embedding_model, model_id)
        self._embedding_model = model_id
        self._build_embedding_client()

    def get_embedding_model(self) -> str:
        return self._embedding_model

    def get_embedding_dimension(self) -> int:
        known_dims = {
            "gemini-embedding-001":                   3072,
            "models/text-embedding-004":               768,
            "text-embedding-004":                      768,
            "models/embedding-001":                    768,
            "models/text-multilingual-embedding-002":  768,
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
        return self._embed([text])

    def embed_documents(self, texts: list[str]) -> EmbeddingResponse:
        if not texts:
            raise ValueError("embed_documents: 'texts' must not be empty.")
        return self._embed(texts)

    # ─────────────────────────────────────────────────────────────
    # Health Check
    # ─────────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        try:
            # Lightweight generation ping
            self.generation_client.invoke(
                [HumanMessage(content="ping")],
                max_output_tokens=1,
            )
            # Lightweight embedding ping
            self.embedding_client.embed_query("ping")

            logger.info(
                "health_check OK | gen=%s emb=%s",
                self._generation_model,
                self._embedding_model,
            )
            return True

        except Exception as exc:
            logger.warning("health_check FAILED | %s", exc)
            return False

    # ─────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> EmbeddingResponse:
        try:
            logger.debug("_embed | model=%s | n_texts=%d", self._embedding_model, len(texts))

            if len(texts) == 1:
                vectors = [self.embedding_client.embed_query(texts[0])]
            else:
                vectors = self.embedding_client.embed_documents(texts)

            dim = len(vectors[0]) if vectors else 0

            logger.info(
                "_embed OK | model=%s | n=%d | dim=%d",
                self._embedding_model, len(vectors), dim,
            )

            return EmbeddingResponse(
                embeddings=vectors,
                model_id=self._embedding_model,
                dimension=dim,
            )

        except Exception as exc:
            logger.error("_embed FAILED | %s", exc)
            raise RuntimeError(f"Gemini embedding error: {exc}") from exc