import numpy as np
from pathlib import Path
from typing import Optional
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

from stores.llm.LLMInterface import LLMInterface
from stores.llm.LLMEnums import HuggingFaceEnums
from stores.llm.schema import Message, GenerationConfig, GenerationResponse, EmbeddingResponse
from core import get_logger, get_settings

logger = get_logger("HuggingFaceProvider:")

class HuggingFaceLLMProvider(LLMInterface):

    def __init__(self, api_key: str) -> None:
        self.client = InferenceClient(api_key=api_key)
        self._generation_model = get_settings().GENERATION_MODEL_ID
        self._embedding_model  = get_settings().EMBEDDING_MODEL_ID

    # --- generation model ---
    def set_generation_model(self, model_id: str) -> None:
        self._ensure_model(model_id)
        self._generation_model = model_id

    def get_generation_model(self) -> str: return self._generation_model

    # --- embedding model ---
    def set_embedding_model(self, model_id: str) -> None:
        self._ensure_model(model_id)
        self._embedding_model = model_id

    def get_embedding_model(self) -> str: return self._embedding_model

    def _ensure_model(self, model_id: str) -> None:
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.utils import RepositoryNotFoundError

        local_dir = Path(get_settings().HF_MODEL_DIR).expanduser().resolve() / model_id.replace("/", "--")

        # Already downloaded — nothing to do
        if local_dir.exists() and any(local_dir.iterdir()):
            logger.debug("Model '%s' already cached at '%s'.", model_id, local_dir)
            return

        # Verify it exists on the Hub before attempting download
        try:
            HfApi().model_info(model_id)
        except RepositoryNotFoundError:
            raise ValueError(f"Model '{model_id}' does not exist on Hugging Face or is private.")

        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(repo_id=model_id, local_dir=local_dir, resume_download=True)
            logger.info("Model '%s' downloaded to '%s'.", model_id, local_dir)
        except Exception as exc:
            logger.warning("Failed to download model '%s': %s", model_id, exc)
            raise RuntimeError(f"Could not download model '{model_id}'.") from exc

    # --- generation ---
    def generate_text(self, messages: list[Message], system_prompt: Optional[str] = None,
                      config: Optional[GenerationConfig] = None) -> GenerationResponse:
        config = config or GenerationConfig()
        api_messages = []
        if system_prompt:
            api_messages.append({"role": HuggingFaceEnums.SYSTEM.value, "content": system_prompt})
        for msg in messages:
            api_messages.append({"role": getattr(HuggingFaceEnums, msg.role.upper()).value, "content": msg.content})
        return self._generate(api_messages, config)

    def _generate(self, api_messages: list[dict], config: GenerationConfig) -> GenerationResponse:
        response = self.client.chat.completions.create(
            model=self._generation_model,
            messages=api_messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stop=config.stop or None,
        )
        choice, usage = response.choices[0], response.usage
        return GenerationResponse(
            content=choice.message.content or "",
            model_id=self._generation_model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            finish_reason=choice.finish_reason,
        )

    # --- embedding ---
    def embed_query(self, text: str) -> EmbeddingResponse:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> EmbeddingResponse:
        if not texts:
            raise ValueError("texts must not be empty.")
        return self._embed(texts)

    def _embed(self, texts) -> EmbeddingResponse:
        input_texts = [texts] if isinstance(texts, str) else texts
        arr = np.array(self.client.feature_extraction(text=input_texts, model=self._embedding_model))

        if arr.ndim == 3:   vectors = arr.mean(axis=1).tolist()   # token-level → mean pool
        elif arr.ndim == 2: vectors = arr.tolist()                 # sentence-level
        elif arr.ndim == 1: vectors = [arr.tolist()]               # single text
        else: raise ValueError(f"Unexpected embedding shape: {arr.shape}")

        return EmbeddingResponse(embeddings=vectors, model_id=self._embedding_model, dimension=len(vectors[0]))

    # --- health ---
    def health_check(self) -> bool:
        try:
            self.client.chat.completions.create(model=self._generation_model, messages=[{"role": "user", "content": "ping"}], max_tokens=1)
            self.client.feature_extraction(text="ping", model=self._embedding_model)
            return True
        except Exception as exc:
            logger.warning("health_check FAILED | %s", exc)
            return False