import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """
    Application settings with validation.
    All settings can be overridden via environment variables.
    """

    # Project Info
    PROJECT_NAME: str = "indxai OS"
    VERSION: str = "1.0.0"

    # AI Model Settings
    TRANSFORMER_MODEL: str = Field(
        default="all-MiniLM-L6-v2", description="Sentence transformer model name"
    )
    LATENT_DIM: int = Field(
        default=24, ge=8, le=256, description="VAE latent dimension"
    )
    EMBEDDING_DIM: int = Field(
        default=384, description="Embedding dimension (must match transformer model)"
    )
    HIDDEN_DIM: int = Field(
        default=128, ge=32, le=512, description="VAE hidden layer dimension"
    )

    # Search Settings
    SEARCH_CACHE_TTL: int = Field(
        default=300, description="Search cache TTL in seconds"
    )
    SEARCH_CACHE_MAX_SIZE: int = Field(
        default=100, description="Maximum search cache entries"
    )
    MAX_SEARCH_RESULTS: int = Field(
        default=5, ge=1, le=20, description="Maximum search results per query"
    )

    # Rate Limiting
    RATE_LIMIT_CHAT: str = Field(
        default="30/minute", description="Rate limit for chat endpoint"
    )
    RATE_LIMIT_TRAIN: str = Field(
        default="5/minute", description="Rate limit for training endpoint"
    )

    # Database Settings
    DATABASE_PATH: str = Field(default="brain.db", description="SQLite database path")

    # Google Search Credentials
    GOOGLE_CSE_ID: Optional[str] = Field(
        default=None, description="Google Custom Search Engine ID"
    )
    GOOGLE_API_KEY: Optional[str] = Field(default=None, description="Google API Key")

    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(default="json", description="Log format: json or console")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()
