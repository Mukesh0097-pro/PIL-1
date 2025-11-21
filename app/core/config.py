import os
from dataclasses import dataclass


@dataclass
class Settings:
    PROJECT_NAME: str = "indxai OS"
    VERSION: str = "1.0.0"
    # Use a tiny model for cloud deployment to stay within free tier RAM
    TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"
    LATENT_DIM: int = 64
    EMBEDDING_DIM: int = 384


settings = Settings()
