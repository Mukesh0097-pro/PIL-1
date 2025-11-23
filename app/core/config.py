import os
from dataclasses import dataclass


@dataclass
class Settings:
    PROJECT_NAME: str = "indxai OS"
    VERSION: str = "1.0.0"

    # AI Model Settings
    TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"
    LATENT_DIM: int = 64
    EMBEDDING_DIM: int = 384

    # --- GOOGLE SEARCH CREDENTIALS ---
    # 1. The ID you just gave me:
    GOOGLE_CSE_ID: str = "502b9f5b609884bf9"

    # 2. The Key you get from Cloud Console (starts with AIza...):
    GOOGLE_API_KEY: str = "AQ.Ab8RN6LyTO7Rta5ob5SyunfSGmaQfJHfhGLS5ZFxSJENFLrWRg"


settings = Settings()
