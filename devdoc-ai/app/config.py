import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_url: str = "http://ollama:11434"
    ollama_model: str = "llama3"
    embedding_model: str = "all-MiniLM-L6-v2"
    data_dir: str = "/data"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 3
    api_key: str = ""
    max_file_size_mb: int = 50

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
