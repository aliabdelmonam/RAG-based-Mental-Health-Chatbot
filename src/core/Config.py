from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
# Resolve the absolute path to the .env file at the project root
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE_PATH = BASE_DIR / ".env"

class Settings(BaseSettings):
    APP_ENV: str = "development"
    GROQ_API_KEY: str
    HF_API_KEY: str = ""
    HF_MODEL_DIR: str = "hugging_face_models"
    GENERATION_MODEL_ID: str = "openai/gpt-oss-safeguard-20b"
    EMBEDDING_MODEL_ID: str

    GENERATION_BACKEND:str
    EMBEDDING_BACKEND:str

    model_config = SettingsConfigDict(

        env_file=ENV_FILE_PATH,
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Instantiate settings once so it can be shared across modules
settings = Settings()

def get_settings() -> Settings:
    return settings


