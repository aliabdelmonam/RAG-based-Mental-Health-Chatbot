from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    APP_ENV:str
    GROQ_API_KEY: str



    class Config:
        env_file = ".env"

def get_settings() -> Settings:
    return Settings()

