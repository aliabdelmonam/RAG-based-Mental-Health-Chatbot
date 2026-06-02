import requests
from typing import Optional

from src.stores.generation import LLMGenerationInterface
from src.stores.LLMEnums import HuggingFaceEnums
from src.stores.schema import Message, GenerationConfig, GenerationResponse
from src.core.logger import get_logger

logger = get_logger("ColabProvider:")


class ColabLLMProvider(LLMGenerationInterface):
    """
    Generation provider that calls a HuggingFace model running on
    Google Colab, exposed via an ngrok tunnel (Flask /generate endpoint).

    Usage
    -----
    provider = ColabLLMProvider(base_url="https://xxxx.ngrok-free.app")
    response = provider.generate_text(messages=[...], system_prompt="...")
    """

    def __init__(self, base_url: str) -> None:
        """
        Args:
            base_url:         The ngrok public URL printed by Colab,
                              e.g. "https://abcd-34-123-45-67.ngrok-free.app"
            generation_model: Label for logging — use the actual model name
                              e.g. "Qwen/Qwen2.5-7B-Instruct"
        """
        # strip trailing slash so we can always do base_url + "/generate"
        self._base_url = base_url.rstrip("/")
        # self._generation_model = generation_model
        logger.info("ColabLLMProvider initialized | url=%s model=%s", self._base_url, self._generation_model)

    # ── Generation Interface ──────────────────────────────────────

    def set_generation_model(self, model_id: str) -> None:
        logger.debug("Generation model label changed: %s → %s", self._generation_model, model_id)
        self._generation_model = model_id

    def get_generation_model(self) -> str:
        return self._generation_model

    def generate_text(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResponse:

        config = config or GenerationConfig()

        # Build message list in the same format the Flask endpoint expects
        api_messages: list[dict] = []

        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            api_messages.append({"role": msg.role, "content": msg.content})

        payload = {
            "messages":       api_messages,
            "max_new_tokens": config.max_tokens,
            "temperature":    config.temperature,
        }

        logger.debug(
            "generate_text | url=%s | messages=%d | temp=%.2f | max_tokens=%d",
            self._base_url, len(api_messages), config.temperature, config.max_tokens,
        )

        try:
            resp = requests.post(
                f"{self._base_url}/generate",
                json=payload,
                timeout=180,   # generation can be slow on free T4
            )
            resp.raise_for_status()
            content = resp.json().get("response", "")

            logger.info("generate_text OK | model=%s | chars=%d", self._generation_model, len(content))

            return GenerationResponse(
                content=content,
                model_id=self._generation_model,
                input_tokens=0,   # Flask endpoint doesn't return token counts
                output_tokens=0,
                finish_reason="stop",
            )

        except requests.exceptions.Timeout:
            logger.error("generate_text FAILED | timeout after 180s")
            raise RuntimeError("Colab generation timed out — is the Colab session still running?")

        except requests.exceptions.RequestException as exc:
            logger.error("generate_text FAILED | %s", exc)
            raise RuntimeError(f"Colab generation error: {exc}") from exc

    # ── Health Check ─────────────────────────────────────────────

    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self._base_url}/health", timeout=10)
            ok = resp.status_code == 200
            if ok:
                logger.info("health_check OK | url=%s", self._base_url)
            else:
                logger.warning("health_check FAILED | status=%d", resp.status_code)
            return ok
        except Exception as exc:
            logger.warning("health_check FAILED | %s", exc)
            return False