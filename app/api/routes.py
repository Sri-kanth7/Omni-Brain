"""
API route definitions for OmniBrain.

Endpoints:
    POST   /upload          Upload and process a PDF document
    POST   /chat            Ask a question about documents
    GET    /documents       List all processed documents
    DELETE /documents/{id}  Delete a specific document
    GET    /health          Health check and system stats
"""

from typing import Any

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Query

from app.core.dependencies import (
    verify_api_key,
    get_document_service,
    get_chat_service,
)
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    DocumentListResponse,
    DocumentInfo,
    ErrorResponse,
    HealthResponse,
    SourceCitation,
    UploadResponse,
)
from app.services.document_service import DocumentService
from app.services.chat_service import ChatService
from app.core.exceptions import OmniBrainError
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


# ═══════════════════════════════════════════════════════════════════
# POST /upload
# ═══════════════════════════════════════════════════════════════════

@router.post(
    "/upload",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Upload and process a PDF document",
    description="Upload a PDF file. The backend extracts text, chunks it, generates embeddings, and indexes it in ChromaDB.",
)
async def upload_document(
    file: UploadFile = File(..., description="PDF file to upload (max 50MB)"),
    document_service: DocumentService = Depends(get_document_service),
    _auth: None = Depends(verify_api_key),
) -> UploadResponse:
    """
    Handle file upload and document processing.

    Reads the uploaded PDF, processes it through the full pipeline
    (extraction → chunking → embedding → indexing), and returns
    the document ID and processing statistics.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    logger.info(f"Upload request: filename='{file.filename}', content_type='{file.content_type}'")

    try:
        file_bytes = await file.read()
        result = await document_service.upload_and_process(
            filename=file.filename,
            file_bytes=file_bytes,
        )
        return UploadResponse(
            document_id=result["document_id"],
            filename=result["filename"],
            pages=result["pages"],
            chunks=result["chunks"],
            status=result["status"],
            message=result["message"],
        )
    except OmniBrainError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except Exception as exc:
        logger.exception("Unexpected error during upload")
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")


# ═══════════════════════════════════════════════════════════════════
# POST /chat
# ═══════════════════════════════════════════════════════════════════

@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Ask a question about the documents",
    description="Submit a question. The system retrieves relevant chunks from indexed documents and generates an answer using Gemini via the LangGraph workflow.",
)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
    _auth: None = Depends(verify_api_key),
) -> ChatResponse:
    """
    Process a user question and return a context-aware answer with citations.

    The RAG pipeline:
    1. Embed the question
    2. Search ChromaDB for relevant chunks
    3. Build context with source citations
    4. Run LangGraph workflow → Gemini generates the answer
    """
    logger.info(f"Chat request: question='{request.question[:100]}...' (top_k={request.top_k})")

    try:
        result = await chat_service.ask(
            question=request.question,
            document_ids=request.document_ids,
            top_k=request.top_k,
            temperature=request.temperature,
        )

        sources = [
            SourceCitation(
                document_id=s["document_id"],
                filename=s["filename"],
                page_number=s["page_number"],
                chunk_index=s["chunk_index"],
                similarity_score=s["similarity_score"],
                snippet=s["snippet"],
            )
            for s in result.get("sources", [])
        ]

        return ChatResponse(
            answer=result["answer"],
            sources=sources,
            processing_time_ms=result["processing_time_ms"],
            model_used=result["model_used"],
        )
    except OmniBrainError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except Exception as exc:
        logger.exception("Unexpected error during chat")
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")


# ═══════════════════════════════════════════════════════════════════
# GET /documents
# ═══════════════════════════════════════════════════════════════════

@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all processed documents",
    description="Returns metadata for all PDFs that have been uploaded and indexed.",
)
async def list_documents(
    document_service: DocumentService = Depends(get_document_service),
    _auth: None = Depends(verify_api_key),
) -> DocumentListResponse:
    """Return a list of all processed documents with their metadata."""
    try:
        documents = await document_service.get_all_documents()
        doc_infos = [
            DocumentInfo(
                document_id=doc["document_id"],
                filename=doc["filename"],
                pages=doc["pages"],
                chunks=doc["chunks"],
                file_size_bytes=doc["file_size_bytes"],
                created_at=doc["created_at"],
                status=doc["status"],
            )
            for doc in documents
        ]
        return DocumentListResponse(documents=doc_infos, total=len(doc_infos))
    except Exception as exc:
        logger.exception("Error listing documents")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {exc}")


# ═══════════════════════════════════════════════════════════════════
# DELETE /documents/{id}
# ═══════════════════════════════════════════════════════════════════

@router.delete(
    "/documents/{document_id}",
    response_model=DeleteResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Delete a processed document",
    description="Removes a document from the vector store, file system, and analytics registry.",
)
async def delete_document(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
    _auth: None = Depends(verify_api_key),
) -> DeleteResponse:
    """Delete a document and all its associated data."""
    try:
        result = await document_service.delete_document(document_id)
        return DeleteResponse(
            document_id=result["document_id"],
            status=result["status"],
            message=result["message"],
        )
    except OmniBrainError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except Exception as exc:
        logger.exception(f"Error deleting document {document_id}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {exc}")


# ═══════════════════════════════════════════════════════════════════
# GET /health
# ═══════════════════════════════════════════════════════════════════

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns system health, version, uptime, and indexing statistics.",
)
async def health_check(
    document_service: DocumentService = Depends(get_document_service),
) -> HealthResponse:
    """Return system health information."""
    from app.services.analytics_service import AnalyticsService
    analytics = AnalyticsService()
    stats = document_service.get_health_stats()

    return HealthResponse(
        status="ok",
        version="1.0.0",
        uptime_seconds=analytics.get_uptime_seconds(),
        documents_indexed=stats["documents_indexed"],
        total_chunks=stats["total_chunks"],
    )