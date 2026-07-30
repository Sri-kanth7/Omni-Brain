"""
FastAPI dependency injection container.
Provides singleton service instances for the application.
"""

from typing import AsyncGenerator

from fastapi import Header, HTTPException

from app.core.config import settings
from app.core.logger import get_logger

from app.services.pdf_service import PDFService
from app.services.analytics_service import AnalyticsService
from app.services.document_service import DocumentService
from app.services.chat_service import ChatService

from app.database.vector_store import VectorStore
from app.database.embeddings import EmbeddingGenerator

from app.ai.gemini_service import GeminiService
from app.ai.langgraph_workflow import LangGraphWorkflow

logger = get_logger(__name__)


# ============================================================
# SINGLETON INSTANCES
# ============================================================

embedding_generator = EmbeddingGenerator()
vector_store = VectorStore()
pdf_service = PDFService()
analytics_service = AnalyticsService()
gemini_service = GeminiService()
langgraph_workflow = LangGraphWorkflow()

document_service = DocumentService(
    pdf_service=pdf_service,
    analytics_service=analytics_service,
    embedding_generator=embedding_generator,
    vector_store=vector_store,
)

chat_service = ChatService()


# ============================================================
# Authentication
# ============================================================

def verify_api_key(x_api_key: str = Header(default=None)) -> None:
    if settings.AUTH_ENABLED:
        if not x_api_key or x_api_key != settings.API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ============================================================
# Dependency Providers
# ============================================================

def get_embedding_generator() -> EmbeddingGenerator:
    return embedding_generator


def get_vector_store() -> VectorStore:
    return vector_store


def get_pdf_service() -> PDFService:
    return pdf_service


def get_analytics_service() -> AnalyticsService:
    return analytics_service


def get_gemini_service() -> GeminiService:
    return gemini_service


def get_langgraph_workflow() -> LangGraphWorkflow:
    return langgraph_workflow


def get_document_service() -> DocumentService:
    return document_service


def get_chat_service() -> ChatService:
    return chat_service


async def get_services() -> AsyncGenerator[dict, None]:
    yield {
        "pdf_service": pdf_service,
        "analytics_service": analytics_service,
        "document_service": document_service,
        "chat_service": chat_service,
        "vector_store": vector_store,
        "embedding_generator": embedding_generator,
        "gemini_service": gemini_service,
        "langgraph_workflow": langgraph_workflow,
    }
