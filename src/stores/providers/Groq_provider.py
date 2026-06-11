from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.messages import  HumanMessage,BaseMessage
from src.stores.generation import LLMGenerationInterface
from src.stores.LLMEnums import GroqEnums
from src.stores.schema import (
    Message,
    GenerationConfig,
    GenerationResponse,
)
from src.core.logger import get_logger
from langchain_core.tools import tool 

logger = get_logger(f'GroqProvider:')

# this is a dummy function tool call, it should be calling police or something
@tool
def crisis_tool(query:str)-> str:
    """
    A tool for handling crisis situations. When invoked, it should trigger an alert to the appropriate support team or provide resources to the user.

    Args:
        query (str): The input query that may indicate a crisis situation.

    Returns:
        str: A response with appropriate resources or an alert message.
    """
    # Placeholder implementation - replace with actual crisis handling logic
    import json

    return json.dumps({
        "crisis_detected": True,
        "resources": [
            "Crisis Text Line: Text HOME to 741741",
            "Suicide Prevention Lifeline: 1-800-273-8255",
            "I Recommend u to go to MR ALWA he is a very very good doctor"
        ]
    })

class GroqLLMProvider(LLMGenerationInterface):
    """
    Groq provider for ultra-fast text generation via LangChain + Groq Cloud API.

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

    DEFAULT_MODEL = "openai/gpt-oss-safeguard-20b"

    # ─────────────────────────────────────────────────────────────
    # Construction
    # ─────────────────────────────────────────────────────────────

    def __init__(
        self,
        api_key: str,
    ) -> None:
        self._api_key = api_key
        self._generation_model: str = None
        logger.info("GroqLLMProvider initialized.")

    def _build_client(self) -> None:
        """Instantiate (or re-instantiate) the LangChain ChatGroq client."""
        if self._generation_model is None:
            logger.warning("_build_client: generation model not set, use set_generation_model() to set one.")
            return
        self.client: ChatGroq = ChatGroq(
            api_key=self._api_key,
            model=self._generation_model,
        )

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
        self._build_client()   # rebuild with new model

    def get_generation_model(self) -> str:
        """Return the currently active Groq model ID."""
        return self._generation_model

    def generate_text(
        self,
        messages: list[BaseMessage],
        config: Optional[GenerationConfig] = None,
        enable_tools: bool = False,
    ) -> GenerationResponse:
        """
        Send a chat-completion request via LangChain + Groq.

        Args:
            messages:      Conversation history (user / assistant turns).
            system_prompt: Optional system-level instruction injected before
                           the conversation history.
            config:        Generation parameters (temperature, max_tokens, stop).
                           Falls back to GenerationConfig defaults if None.

        Returns:
            GenerationResponse with generated text and token-usage metadata.

        Raises:
            RuntimeError: wraps any exception so callers get a clean exception.
        """
        if config is None:
            config = GenerationConfig()

        # ── Map internal Message objects → LangChain message types ──
        # _role_map = {
            # "system":    SystemMessage,
            # "user":      HumanMessage,
            # "assistant": AIMessage,
        # }

        # lc_messages = []

        # if system_prompt:
            # lc_messages.append(SystemMessage(content=system_prompt))

        # for msg in messages:
            # msg_class = _role_map.get(msg.role, HumanMessage)
            # lc_messages.append(msg_class(content=msg.content))

        try:
            logger.debug(
                "generate_text | model=%s | messages=%d | temp=%.2f | max_new_tokens=%d",
                self._generation_model,
                len(messages),
                config.temperature,
                config.max_new_tokens,
            )

            # Bind generation config at call time
            bound_client = self.client.bind(
                temperature=config.temperature,
                max_tokens=config.max_new_tokens,
                stop=config.stop if config.stop else None,
            )
            model = bound_client
            if enable_tools:
                model = bound_client.bind_tools([crisis_tool])  # Register the crisis tool with the client
            response = model.invoke(messages)

            # ── Extract usage metadata ──────────────────────────────
            usage = response.response_metadata.get("token_usage", {})
            input_tokens  = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            finish_reason = response.response_metadata.get("finish_reason", "unknown")
            model_id      = response.response_metadata.get("model", self._generation_model)

            logger.info(
                "generate_text OK | model=%s | in=%d out=%d tokens | finish=%s",
                model_id,
                input_tokens,
                output_tokens,
                finish_reason,
            )
            if response.tool_calls:
                logger.info(f"Tool calls: {response.tool_calls}")
                return GenerationResponse(
                    content=response.tool_calls,  # carry tool_calls as content
                    model_id=model_id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    finish_reason="tool_call",
                )

            return GenerationResponse(
                content=response.content or "",
                model_id=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
            )
        except Exception as exc:
            logger.error("generate_text FAILED | %s", exc)
            raise RuntimeError(f"Groq generation error: {exc}") from exc

    # ─────────────────────────────────────────────────────────────
    # Health Check
    # ─────────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """
        Verify the Groq API is reachable and the configured model is available.

        Returns:
            True  — API responded successfully.
            False — any connectivity, auth, or model error.
        """
        try:
            self.client.bind(max_tokens=1).invoke(
                [HumanMessage(content="ping")]
            )
            logger.info("health_check OK | model=%s", self._generation_model)
            return True
        except Exception as exc:
            logger.warning("health_check FAILED | %s", exc)
            return False