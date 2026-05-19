from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the absolute path to the .env file at the project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE_PATH = BASE_DIR / ".env"

class Settings(BaseSettings):
    APP_ENV: str = "development"
    GROQ_API_KEY: str
    GENERATION_MODEL_ID: str = "openai/gpt-oss-safeguard-20b"

    model_config = SettingsConfigDict(

        env_file=ENV_FILE_PATH,
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Instantiate settings once so it can be shared across modules
settings = Settings()

def get_settings() -> Settings:
    return settings


