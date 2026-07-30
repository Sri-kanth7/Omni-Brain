"""
Application configuration.
Loads environment variables and provides typed settings.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings:
    """Centralized application settings loaded from environment."""

    # ── Project ──────────────────────────────────────────────────────
    APP_NAME: str = "OmniBrain"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "AI-powered document Q&A system"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # ── Server ───────────────────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"]

    # ── Authentication (placeholder) ─────────────────────────────────
    API_KEY: Optional[str] = os.getenv("API_KEY")
    AUTH_ENABLED: bool = os.getenv("AUTH_ENABLED", "false").lower() == "true"

    # ── Google Gemini ────────────────────────────────────────────────
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "GEMINI_MODEL=gemini-1.5-flash"
    GEMINI_TEMPERATURE: float = 0.2
    GEMINI_MAX_TOKENS: int = 4096

    # ── Embeddings ───────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # ── ChromaDB ─────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "vector_store")
    CHROMA_COLLECTION_NAME: str = "omnibrain_docs"

    # ── Upload ───────────────────────────────────────────────────────
    UPLOAD_DIR: str = str(BASE_DIR / "uploads")
    REPORTS_DIR: str = str(BASE_DIR / "reports")
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: set[str] = {".pdf"}

    # ── Chunking ─────────────────────────────────────────────────────
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # ── Retrieval ────────────────────────────────────────────────────
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.5

    # ── Logging ──────────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "json"


settings = Settings()

# ── Ensure required directories exist ──────────────────────────────
for _dir in [settings.UPLOAD_DIR, settings.CHROMA_PERSIST_DIR, settings.REPORTS_DIR]:
    Path(_dir).mkdir(parents=True, exist_ok=True)
