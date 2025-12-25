"""
Dependency Injection for FastAPI.
Thread-safe singleton pattern for the AI engine.
"""

import threading
from functools import lru_cache
from typing import Optional

# Thread lock for singleton initialization
_lock = threading.Lock()
_engine_instance: Optional["IndxAI_OS"] = None


def get_engine():
    """
    Get or create the singleton engine instance.
    Thread-safe lazy initialization.
    """
    global _engine_instance

    if _engine_instance is None:
        with _lock:
            # Double-check locking pattern
            if _engine_instance is None:
                from app.core.engine import IndxAI_OS

                _engine_instance = IndxAI_OS()

    return _engine_instance


def reset_engine():
    """
    Reset the engine instance (useful for testing).
    """
    global _engine_instance
    with _lock:
        _engine_instance = None


@lru_cache()
def get_settings():
    """
    Cached settings instance.
    """
    from app.core.config import Settings

    return Settings()
