from pathlib import Path
from typing import Any
from dotenv import load_dotenv

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the absolute path to the .env file at the project root
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE_PATH = BASE_DIR / ".env"

# Load the environment variables into the OS environment so LangChain can see them
load_dotenv(ENV_FILE_PATH)


class Settings(BaseSettings):
    APP_ENV: str = "development"

    GROQ_API_KEY: str
    HF_API_KEY: str = ""
    HF_MODEL_DIR: str = "hugging_face_models"
    GENERATION_MODEL_ID_PRIMARY: str = "openai/gpt-oss-safeguard-20b"
    EMBEDDING_MODEL_ID_PRIMARY: str

    GENERATION_BACKEND_PRIMAY: str
    EMBEDDING_BACKEND_PRIMARY: str

    GENERATION_BACKEND_FALLBACK:str
    EMBEDDING_BACKEND_FALLBACK:str

    GENERATION_MODEL_ID_FALLBACK:str
    EMBEDDING_MODEL_ID_FALLBACK:str

    VECTORDB_BACKEND: str
    QDRANT_PATH: str
    QDRANT_URL: str

    # NOTE: in some .env setups this can be an empty string (QDRANT_IN_MEMORY=),
    # which breaks bool parsing in Pydantic v2.
    QDRANT_IN_MEMORY: str
    QDRANT_API_KEY: str

    COLAB_NGROK_URL: str
    GEMINI_API_KEY:str
    COHERE_API_KEY:str

    lang_detection_model:str
    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Instantiate settings once so it can be shared across modules
settings = Settings()


def get_settings() -> Settings:
    return settings



