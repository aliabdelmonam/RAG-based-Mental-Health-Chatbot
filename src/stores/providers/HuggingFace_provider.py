import time
import numpy as np
from pathlib import Path
from typing import Optional
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

from src.stores.LLMInterface import LLMInterface
from src.stores.LLMEnums import HuggingFaceEnums
from src.stores.schema import Message, GenerationConfig, GenerationResponse, EmbeddingResponse
from src.core import get_logger, get_settings


logger = get_logger("HuggingFaceProvider:")
settings = get_settings()


class HuggingFaceLLMProvider(LLMInterface):

    def __init__(self, api_key: str) -> None:
        self.client = InferenceClient(api_key=api_key)
        self._generation_model = settings.GENERATION_MODEL_ID  # always stays a string
        self._embedding_model = settings.EMBEDDING_MODEL_ID
        self._local_model = None
        self._local_tokenizer = None
        self._local_embedding_model = None

        # load models from local path on startup
        gen_path = Path(settings.HF_MODEL_DIR).expanduser().resolve() / self._generation_model.replace("/", "--")
        emb_path = Path(settings.HF_MODEL_DIR).expanduser().resolve() / self._embedding_model.replace("/", "--")

        # self._ensure_model(self._generation_model)
        # self._load_model(local_path=gen_path)

        # self._ensure_model(self._embedding_model)
        # self._load_embedding_model(local_path=emb_path)

        logger.info(
            "HuggingFaceLLMProvider initialized. | gen_model=%s emb_model=%s",
            self._generation_model,
            self._embedding_model,
        )


    # --- generation model ---
    def set_generation_model(self, model_id: str) -> None:
        logger.debug("Generation model changed: %s -> %s", self._generation_model, model_id)
        model_path = Path(settings.HF_MODEL_DIR).expanduser().resolve() / model_id.replace("/", "--")
        self._ensure_model(model_id)
        self._generation_model = model_id
        self._load_model(local_path=model_path)

    def get_generation_model(self) -> str:
        return self._generation_model

    # --- embedding model ---
    def set_embedding_model(self, model_id: str) -> None:
        logger.debug("Embedding model changed: %s -> %s", self._embedding_model, model_id)
        model_path = Path(settings.HF_MODEL_DIR).expanduser().resolve() / model_id.replace("/", "--")
        self._ensure_model(model_id)
        self._embedding_model = model_id
        self._load_embedding_model(local_path=model_path)

    def get_embedding_model(self) -> str:
        return self._embedding_model

    def _load_embedding_model(self, local_path: Path) -> None:
        logger.info("Loading local embedding model from: %s", local_path)
        try:
            from sentence_transformers import SentenceTransformer
            self._local_embedding_model = SentenceTransformer(str(local_path))
            logger.info("Embedding model loaded successfully.")
        except Exception as exc:
            logger.critical("Failed to load embedding model from '%s'. | Error: %s", local_path, exc)
            raise

    def get_embedding_dimension(self) -> int:
        cached = getattr(self, "_embedding_dimension", None)
        if isinstance(cached, int) and cached > 0:
            return cached

        if self._local_embedding_model is None:
            raise RuntimeError("Local embedding model is not loaded.")

        dim = self._local_embedding_model.get_sentence_embedding_dimension()
        if not dim:
            raise RuntimeError(f"Could not determine embedding dimension for model '{self._embedding_model}'.")

        self._embedding_dimension = dim
        return dim


    def _ensure_model(self, model_id: str) -> None:
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.utils import RepositoryNotFoundError

        local_dir = Path(settings.HF_MODEL_DIR).expanduser().resolve() / model_id.replace("/", "--")

        # Already downloaded — nothing to do
        if local_dir.exists() and any(local_dir.iterdir()):
            logger.debug("Model '%s' already cached at '%s'.", model_id, local_dir)
            return

        # Verify it exists on the Hub before attempting download.
        # If this fails (private/gated model, missing auth token, etc.),
        # we log and continue so the app can still start.
        try:
            HfApi().model_info(model_id)
        except RepositoryNotFoundError as exc:
            logger.warning(
                "_ensure_model: model_info failed (not found/private?) | model=%s | err=%s",
                model_id,
                exc,
            )
            return
        except Exception as exc:
            logger.warning(
                "_ensure_model: model_info failed | model=%s | err=%s",
                model_id,
                exc,
            )
            return


        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(repo_id=model_id, local_dir=local_dir, resume_download=True)
            logger.info("Starting downloading model...")
            logger.info("Model '%s' downloaded to '%s'.", model_id, local_dir)
        except Exception as exc:
            logger.warning(
                "_ensure_model: snapshot_download failed | model=%s | err=%s",
                model_id,
                exc,
            )
            return

    # loading model
    def _load_model(self, local_path: Path) -> None:
        logger.info("Initializing local inference execution client from directory: %s", local_path)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._local_tokenizer = AutoTokenizer.from_pretrained(local_path)
            self._local_model = AutoModelForCausalLM.from_pretrained(local_path, device_map="auto")
            logger.info("Model weights and configs successfully loaded into local execution context.")
        except Exception as exc:
            logger.critical("Failed to load model into memory from path '%s'. | Error: %s", local_path, exc)
            raise exc

    
    # --- generation ---
    def generate_text(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResponse:
        config = config or GenerationConfig()
        api_messages: list[dict] = []

        if system_prompt:
            api_messages.append({"role": HuggingFaceEnums.SYSTEM.value, "content": system_prompt})

        for msg in messages:
            api_messages.append({"role": getattr(HuggingFaceEnums, msg.role.upper()).value, "content": msg.content})


        logger.debug(
            "generate_text | model=%s | messages=%d | has_system=%s | temp=%.2f | max_tokens=%d",
            self._generation_model,
            len(api_messages),
            bool(system_prompt),
            config.temperature,
            config.max_tokens,
        )

        try:
            return self._generate(api_messages, config)
        except Exception as exc:
            logger.warning(
                "generate_text FAILED | model=%s | err=%s",
                self._generation_model,
                exc,
            )
            raise


    def _generate(self, api_messages: list[dict], config: GenerationConfig) -> GenerationResponse:
        import torch
        logger.debug("_generate | model=%s", self._generation_model)

        tokenized = self._local_tokenizer.apply_chat_template(
            api_messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,  # force plain tensor, not BatchEncoding
        )

        # apply_chat_template with return_dict=False always returns a plain tensor
        input_ids = tokenized.to(self._local_model.device)
        input_token_count = input_ids.shape[-1]

        with torch.no_grad():
            output_ids = self._local_model.generate(
                input_ids,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                do_sample=config.temperature > 0,
                pad_token_id=self._local_tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_token_count:]
        content = self._local_tokenizer.decode(new_tokens, skip_special_tokens=True)
        output_token_count = len(new_tokens)

        logger.info(
            "generate_text OK | model=%s | in=%d out=%d tokens",
            self._generation_model, input_token_count, output_token_count,
        )
        return GenerationResponse(
            content=content,
            model_id=self._generation_model,
            input_tokens=input_token_count,
            output_tokens=output_token_count,
            finish_reason="stop",
        )


    # --- embedding ---
    def embed_query(self, text: str) -> EmbeddingResponse:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> EmbeddingResponse:
        if not texts:
            raise ValueError("embed_documents: 'texts' must not be empty.")

        logger.debug(
            "embed_documents | model=%s | n_texts=%d",
            self._embedding_model,
            len(texts),
        )

        try:
            return self._embed(texts)
        except Exception as exc:
            logger.warning(
                "embed_documents FAILED | model=%s | err=%s",
                self._embedding_model,
                exc,
            )
            raise

    def _embed(self, texts) -> EmbeddingResponse:
        if self._local_embedding_model is None:
            raise RuntimeError("Local embedding model is not loaded.")

        input_texts = [texts] if isinstance(texts, str) else texts
        logger.debug("_embed | model=%s | n_texts=%d", self._embedding_model, len(input_texts))

        vectors = self._local_embedding_model.encode(input_texts, convert_to_numpy=True).tolist()
        dim = len(vectors[0]) if vectors else 0

        logger.info("_embed OK | model=%s | n=%d | dim=%d", self._embedding_model, len(vectors), dim)
        return EmbeddingResponse(embeddings=vectors, model_id=self._embedding_model, dimension=dim)

    # --- health ---
    def health_check(self) -> bool:
        logger.debug("health_check | gen_model=%s emb_model=%s", self._generation_model, self._embedding_model)
        try:
            # check local model is loaded
            if getattr(self, "_local_model", None) is None or getattr(self, "_local_tokenizer", None) is None:
                raise RuntimeError("Local generation model is not loaded.")

            # check local embedding model is loaded
            if getattr(self, "_local_embedding_model", None) is None:
                raise RuntimeError("Local embedding model is not loaded.")

            logger.info("health_check OK | gen=%s emb=%s", self._generation_model, self._embedding_model)
            return True
        except Exception as exc:
            logger.warning("health_check FAILED | gen=%s emb=%s | err=%s", self._generation_model, self._embedding_model, exc)
            return False
