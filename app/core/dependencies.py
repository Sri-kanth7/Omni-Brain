"""
FastAPI dependency injection container.
Provides clean, testable dependencies for route handlers.
"""

from typing import AsyncGenerator

from fastapi import Header, HTTPException, Request

from app.core.config import settings
from app.core.logger import get_logger
from app.services.document_service import DocumentService
from app.services.pdf_service import PDFService
from app.services.analytics_service import AnalyticsService
from app.services.chat_service import ChatService
from app.database.vector_store import VectorStore
from app.database.embeddings import EmbeddingGenerator
from app.ai.gemini_service import GeminiService
from app.ai.langgraph_workflow import LangGraphWorkflow

logger = get_logger(__name__)


def verify_api_key(x_api_key: str = Header(default=None)) -> None:
    """Dependency: validate API key if authentication is enabled."""
    if settings.AUTH_ENABLED:
        if not x_api_key or x_api_key != settings.API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")


def get_embedding_generator() -> EmbeddingGenerator:
    """Provide a singleton EmbeddingGenerator."""
    return EmbeddingGenerator()


def get_vector_store() -> VectorStore:
    """Provide a singleton VectorStore."""
    return VectorStore()


def get_pdf_service() -> PDFService:
    """Provide a singleton PDFService."""
    return PDFService()


def get_analytics_service() -> AnalyticsService:
    """Provide a singleton AnalyticsService."""
    return AnalyticsService()


def get_gemini_service() -> GeminiService:
    """Provide a singleton GeminiService."""
    return GeminiService()


def get_langgraph_workflow() -> LangGraphWorkflow:
    """Provide a singleton LangGraphWorkflow."""
    return LangGraphWorkflow()


def get_document_service() -> DocumentService:
    """Provide a singleton DocumentService."""
    return DocumentService()


def get_chat_service() -> ChatService:
    """Provide a singleton ChatService."""
    return ChatService()


async def get_services() -> AsyncGenerator[dict, None]:
    """
    Composite dependency that yields all services.
    Useful for routes that need multiple dependencies.
    """
    yield {
        "pdf_service": get_pdf_service(),
        "analytics_service": get_analytics_service(),
        "document_service": get_document_service(),
        "chat_service": get_chat_service(),
        "vector_store": get_vector_store(),
        "embedding_generator": get_embedding_generator(),
        "gemini_service": get_gemini_service(),
        "langgraph_workflow": get_langgraph_workflow(),
    }