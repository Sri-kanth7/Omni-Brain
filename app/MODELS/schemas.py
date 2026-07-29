"""
Pydantic models for API request/response validation.
All models include type hints, docstrings, and default values.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Upload ─────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Response returned after a successful PDF upload and processing."""

    document_id: str = Field(..., description="Unique identifier for the uploaded document")
    filename: str = Field(..., description="Original filename")
    pages: int = Field(..., description="Number of pages processed")
    chunks: int = Field(..., description="Number of text chunks created")
    status: str = Field(default="success", description="Processing status")
    message: str = Field(default="Document processed successfully", description="Human-readable message")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="ISO timestamp")


class ErrorResponse(BaseModel):
    """Standard error response payload."""

    detail: str = Field(..., description="Error description")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="ISO timestamp")


# ── Chat / Q&A ─────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request body for the chat / Q&A endpoint."""

    question: str = Field(..., min_length=1, max_length=4096, description="User's question about the documents")
    document_ids: Optional[list[str]] = Field(
        default=None,
        description="Optional list of document IDs to restrict search scope",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of relevant chunks to retrieve")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Override model temperature")


class SourceCitation(BaseModel):
    """Citation for a source chunk used in the answer."""

    document_id: str = Field(..., description="Source document ID")
    filename: str = Field(..., description="Source filename")
    page_number: int = Field(..., description="Page number the chunk came from")
    chunk_index: int = Field(..., description="Chunk index within the document")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Retrieval similarity score")
    snippet: str = Field(..., max_length=500, description="Short text excerpt")


class ChatResponse(BaseModel):
    """Response returned after processing a user's question."""

    answer: str = Field(..., description="AI-generated answer")
    sources: list[SourceCitation] = Field(default_factory=list, description="Source citations")
    processing_time_ms: int = Field(..., description="Total processing time in milliseconds")
    model_used: str = Field(..., description="AI model that generated the answer")


# ── Document Management ────────────────────────────────────────────

class DocumentInfo(BaseModel):
    """Metadata about a processed document."""

    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    pages: int = Field(..., description="Number of pages")
    chunks: int = Field(..., description="Number of chunks")
    file_size_bytes: int = Field(..., description="File size in bytes")
    created_at: str = Field(..., description="Upload timestamp (ISO)")
    status: str = Field(..., description="Processing status")


class DocumentListResponse(BaseModel):
    """Response listing all processed documents."""

    documents: list[DocumentInfo] = Field(default_factory=list, description="List of documents")
    total: int = Field(..., description="Total number of documents")


class DeleteResponse(BaseModel):
    """Response after deleting a document."""

    document_id: str = Field(..., description="ID of the deleted document")
    status: str = Field(default="deleted", description="Deletion status")
    message: str = Field(..., description="Human-readable message")


# ── Health ─────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="ok", description="Service status")
    version: str = Field(..., description="Application version")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")
    documents_indexed: int = Field(..., description="Number of documents in the vector store")
    total_chunks: int = Field(..., description="Total chunks indexed")