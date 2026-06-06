import os
import torch
from pathlib import Path
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import GenerationConfig as HFGenerationConfig
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.embeddings import Embeddings
from src.stores.LLMInterface import LLMInterface
from src.stores.schema import Message, GenerationConfig, GenerationResponse, EmbeddingResponse
from src.core import get_logger, get_settings

logger = get_logger("HuggingFaceProvider:")
settings = get_settings()


class HuggingFaceLLMProvider(LLMInterface,Embeddings):

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._generation_model: Optional[str] = None
        self._embedding_model: Optional[str] = None

        self._lc_generation_pipeline: Optional[HuggingFacePipeline] = None
        self._lc_chat_model: Optional[ChatHuggingFace] = None
        self._lc_embedding_pipeline: Optional[HuggingFaceEmbeddings] = None
        self._cached_config: Optional[GenerationConfig] = None
        self._embedding_dimension: Optional[int] = None

    # ── Generation model management ───────────────────────────────

    def set_generation_model(self, model_id: str) -> None:
        self._generation_model = model_id
        self._lc_generation_pipeline = None
        self._lc_chat_model = None
        self._cached_config = None
        self._ensure_model(model_id)

    def get_generation_model(self) -> str:
        return self._generation_model

    def _build_generation_pipeline(self, config: GenerationConfig) -> None:
        local_path = self._local_path(self._generation_model)

        # ✅ Determine device explicitly — never let "auto" silently pick CPU
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info("Building pipeline | model=%s | device=%s", self._generation_model, device)

        if device == "cpu":
            logger.warning(
                "No CUDA device found — generation will be slow. "
                "Consider switching to Groq or Cohere provider."
            )

        tokenizer = AutoTokenizer.from_pretrained(str(local_path))

        model = AutoModelForCausalLM.from_pretrained(
            str(local_path),
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,  # ✅ float16 needs CUDA
            device_map=device,
        )

        # ✅ All generation params in ONE place — eliminates both warnings
        model.generation_config = HFGenerationConfig(
            max_new_tokens=config.max_new_tokens, 
            temperature=config.temperature if config.temperature > 0 else 1.0,
            do_sample=config.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

        # ✅ pipeline() gets NO generation kwargs — they live in model.generation_config
        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,
            device_map=device,
        )

        self._lc_generation_pipeline = HuggingFacePipeline(pipeline=hf_pipeline)
        self._lc_chat_model = ChatHuggingFace(llm=self._lc_generation_pipeline)
        self._cached_config = config
        logger.info("Pipeline ready | model=%s | device=%s", self._generation_model, device)

    def _get_chat_model(self, config: GenerationConfig) -> ChatHuggingFace:
        if self._lc_chat_model is None or self._cached_config != config:
            self._build_generation_pipeline(config)
        return self._lc_chat_model

    # ── Embedding model management ────────────────────────────────

    def set_embedding_model(self, model_id: str) -> None:
        self._embedding_model = model_id
        self._lc_embedding_pipeline = None
        self._embedding_dimension = None
        self._ensure_model(model_id)

    def get_embedding_model(self) -> str:
        return self._embedding_model

    def _get_embedding_pipeline(self) -> HuggingFaceEmbeddings:
        if self._lc_embedding_pipeline is None:
            local_path = self._local_path(self._embedding_model)
            logger.info("Building embedding pipeline from: %s", local_path)
            self._lc_embedding_pipeline = HuggingFaceEmbeddings(
                model_name=str(local_path),
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._lc_embedding_pipeline

    def get_embedding_dimension(self) -> int:
        if self._embedding_dimension:
            return self._embedding_dimension
        probe = self._get_embedding_pipeline().embed_query("probe")
        self._embedding_dimension = len(probe)
        return self._embedding_dimension

    # ── Generation ────────────────────────────────────────────────

    def generate_text(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResponse:
        config = config or GenerationConfig()
        chat_model = self._get_chat_model(config)

        _role_map = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}

        lc_messages: list[BaseMessage] = []
        if system_prompt:
            lc_messages.append(SystemMessage(content=system_prompt))
        for msg in messages:
            lc_messages.append(_role_map.get(msg.role, HumanMessage)(content=msg.content))

        try:
            response = chat_model.invoke(lc_messages)
            content = response.content.strip()
            logger.info("generate_text OK | model=%s | out_chars=%d", self._generation_model, len(content))
            return GenerationResponse(
                content=content,
                model_id=self._generation_model,
                input_tokens=0,
                output_tokens=0,
                finish_reason="stop",
            )
        except Exception as exc:
            logger.warning("generate_text FAILED | model=%s | err=%s", self._generation_model, exc)
            raise RuntimeError(f"HuggingFace generation error: {exc}") from exc

    # ── Embedding ─────────────────────────────────────────────────

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text]).embeddings[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("embed_documents: 'texts' must not be empty.")
        return self._embed(texts).embeddings 

    def _embed(self, texts: list[str]) -> EmbeddingResponse:
        emb_pipeline = self._get_embedding_pipeline()
        try:
            vectors = (
                [emb_pipeline.embed_query(texts[0])]
                if len(texts) == 1
                else emb_pipeline.embed_documents(texts)
            )
            dim = len(vectors[0]) if vectors else 0
            if not self._embedding_dimension and dim:
                self._embedding_dimension = dim
            return EmbeddingResponse(embeddings=vectors, model_id=self._embedding_model, dimension=dim)
        except Exception as exc:
            logger.error("_embed FAILED | model=%s | err=%s", self._embedding_model, exc)
            raise RuntimeError(f"HuggingFace embedding error: {exc}") from exc

    # ── Health check ──────────────────────────────────────────────

    def health_check(self) -> bool:
        try:
            self._get_embedding_pipeline().embed_query("ping")
            return True
        except Exception as exc:
            logger.warning("health_check FAILED | err=%s", exc)
            return False

    # ── Helpers ───────────────────────────────────────────────────

    def _local_path(self, model_id: str) -> Path:
        return Path(settings.HF_MODEL_DIR).expanduser().resolve() / model_id.replace("/", "--")

    def _ensure_model(self, model_id: str) -> None:
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.utils import RepositoryNotFoundError
        from tqdm import tqdm
        import shutil

        local_dir = self._local_path(model_id)
        sentinel = local_dir / ".download_complete"

        if sentinel.exists():
            logger.debug("Model '%s' already cached.", model_id)
            return

        try:
            model_info = HfApi().model_info(model_id, files_metadata=True)
        except RepositoryNotFoundError as exc:
            raise ValueError(f"Model '{model_id}' not found on HuggingFace Hub.") from exc
        except Exception as exc:
            raise RuntimeError(f"Could not reach HuggingFace Hub: {exc}") from exc

        if local_dir.exists():
            logger.warning("Incomplete download found for '%s', wiping.", model_id)
            shutil.rmtree(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        siblings = model_info.siblings or []
        total_bytes = sum(f.size or 0 for f in siblings)
        logger.info("Downloading '%s' — %d files, %.1f GB", model_id, len(siblings), total_bytes / 1e9)

        try:
            with tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024,
                      desc=model_id.split("/")[-1]) as pbar:
                for repo_file in siblings:
                    hf_hub_download(
                        repo_id=model_id,
                        filename=repo_file.rfilename,
                        local_dir=str(local_dir),
                        resume_download=False,
                    )
                    pbar.update(repo_file.size or 0)

            sentinel.touch()
            logger.info("Model '%s' fully downloaded to '%s'.", model_id, local_dir)

        except Exception as exc:
            shutil.rmtree(local_dir, ignore_errors=True)
            raise RuntimeError(f"Download failed for '{model_id}': {exc}") from exc