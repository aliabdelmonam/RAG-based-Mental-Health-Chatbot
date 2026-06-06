from typing import Optional
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.embeddings import Embeddings

from src.stores.LLMInterface import LLMInterface
from src.stores.schema import Message, GenerationConfig, GenerationResponse, EmbeddingResponse
from src.core.logger import get_logger
from src.stores.providers import crisis_tool

logger = get_logger(__name__)


class CohereLLMProvider(LLMInterface, Embeddings):

    _QUERY_INPUT_TYPE    = "search_query"
    _DOCUMENT_INPUT_TYPE = "search_document"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._generation_model = None
        self._embedding_model  = None
        logger.info("CohereLLMProvider initialized.")

    def _build_generation_client(self) -> None:
        if self._generation_model is None:
            logger.warning("generation model not set.")
            return
        self.generation_client = ChatCohere(
            cohere_api_key=self._api_key,
            model=self._generation_model,
        )

    def _build_embedding_client(self) -> None:
        if self._embedding_model is None:
            logger.warning("embedding model not set.")
            return
        self.embedding_client = CohereEmbeddings(
            cohere_api_key=self._api_key,
            model=self._embedding_model,
        )

    # ── Generation model management ───────────────────────────────

    def set_generation_model(self, model_id: str) -> None:
        logger.debug("Generation model changed: %s → %s", self._generation_model, model_id)
        self._generation_model = model_id
        self._build_generation_client()

    def get_generation_model(self) -> str:
        return self._generation_model

    def generate_text(
        self,
        messages: list,
        config: Optional[GenerationConfig] = None,
        enable_tools: bool = False,
    ) -> GenerationResponse:
        if not messages:
            raise ValueError("generate_text: 'messages' must not be empty.")
        if config is None:
            config = GenerationConfig()

        # ── Normalize messages ────────────────────────────────────
        _role_map = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
        lc_messages: list[BaseMessage] = []
        for msg in messages:
            if isinstance(msg, BaseMessage):
                lc_messages.append(msg)
            else:
                lc_messages.append(_role_map.get(msg.role, HumanMessage)(content=msg.content))

        try:
            logger.debug(
                "generate_text | model=%s | messages=%d | temp=%.2f | max_tokens=%d",
                self._generation_model, len(lc_messages),
                config.temperature, config.max_new_tokens,
            )

            model = self.generation_client
            if enable_tools:
                model = self.generation_client.bind_tools([crisis_tool])

            response = model.invoke(
                lc_messages,
                temperature=config.temperature,
                max_tokens=config.max_new_tokens,
                **({"stop_sequences": config.stop} if config.stop else {}),
            )

            usage         = response.response_metadata.get("token_count", {})
            input_tokens  = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("response_tokens", 0)
            finish_reason = response.response_metadata.get("finish_reason", "stop")

            logger.info(
                "generate_text OK | model=%s | in=%d out=%d tokens | finish=%s",
                self._generation_model, input_tokens, output_tokens, finish_reason,
            )

            if response.tool_calls:
                logger.info("Tool calls: %s", response.tool_calls)
                return GenerationResponse(
                    content=response.tool_calls,
                    model_id=self._generation_model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    finish_reason="tool_call",
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

    # ── Embedding model management ────────────────────────────────

    def set_embedding_model(self, model_id: str) -> None:
        logger.debug("Embedding model changed: %s → %s", self._embedding_model, model_id)
        self._embedding_model = model_id
        self._build_embedding_client()

    def get_embedding_model(self) -> str:
        return self._embedding_model

    def get_embedding_dimension(self) -> int:
        known_dims = {
            "embed-english-v3.0":             1024,
            "embed-multilingual-v3.0":        1024,
            "embed-english-light-v3.0":        384,
            "embed-multilingual-light-v3.0":   384,
            "embed-english-v2.0":             4096,
            "embed-english-light-v2.0":       1024,
            "embed-multilingual-v2.0":         768,
        }
        if self._embedding_model in known_dims:
            return known_dims[self._embedding_model]
        probe = self.embed_query("probe")
        return len(probe)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], input_type=self._QUERY_INPUT_TYPE)[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("embed_documents: 'texts' must not be empty.")
        return self._embed(texts, input_type=self._DOCUMENT_INPUT_TYPE)

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        try:
            logger.debug("_embed | model=%s | input_type=%s | n=%d", self._embedding_model, input_type, len(texts))
            typed_client = CohereEmbeddings(
                cohere_api_key=self._api_key,
                model=self._embedding_model,
                input_type=input_type,
            )
            vectors = (
                [typed_client.embed_query(texts[0])]
                if len(texts) == 1
                else typed_client.embed_documents(texts)
            )
            logger.info("_embed OK | model=%s | n=%d | dim=%d", self._embedding_model, len(vectors), len(vectors[0]))
            return vectors
        except Exception as exc:
            logger.error("_embed FAILED | %s", exc)
            raise RuntimeError(f"Cohere embedding error: {exc}") from exc

    # ── Health check ──────────────────────────────────────────────

    def health_check(self) -> bool:
        try:
            self.generation_client.invoke([HumanMessage(content="ping")], max_tokens=1)
            self.embedding_client.embed_query("ping")
            logger.info("health_check OK | gen=%s emb=%s", self._generation_model, self._embedding_model)
            return True
        except Exception as exc:
            logger.warning("health_check FAILED | %s", exc)
            return False