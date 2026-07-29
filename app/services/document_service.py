"""
Document management service.
Orchestrates PDF processing, embedding, indexing, and metadata registration.
"""

import uuid
from pathlib import Path
from typing import Any, Optional

from app.core.config import settings
from app.core.exceptions import (
    DocumentNotFoundError,
    FileTooLargeError,
    InvalidFileTypeError,
    ProcessingError,
)
from app.core.logger import get_logger
from app.database.embeddings import EmbeddingGenerator
from app.database.vector_store import VectorStore
from app.services.pdf_service import PDFService
from app.services.analytics_service import AnalyticsService

logger = get_logger(__name__)


class DocumentService:
    """
    High-level service that orchestrates the full document ingestion pipeline:

    1. Validate and save uploaded file
    2. Extract text via PDFService
    3. Preprocess and chunk via PDFService
    4. Validate chunks via AnalyticsService
    5. Generate embeddings via EmbeddingGenerator
    6. Store in VectorStore (ChromaDB)
    7. Register metadata via AnalyticsService
    """

    def __init__(
        self,
        pdf_service: Optional[PDFService] = None,
        analytics_service: Optional[AnalyticsService] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        vector_store: Optional[VectorStore] = None,
    ) -> None:
        self._pdf_service = pdf_service or PDFService()
        self._analytics_service = analytics_service or AnalyticsService()
        self._embedding_generator = embedding_generator or EmbeddingGenerator()
        self._vector_store = vector_store or VectorStore()

    async def upload_and_process(self, filename: str, file_bytes: bytes) -> dict[str, Any]:
        """
        Validate, save, and process an uploaded PDF file.

        Args:
            filename: Original filename from the upload.
            file_bytes: Raw file content.

        Returns:
            Upload response dict with document_id, pages, chunks, etc.

        Raises:
            InvalidFileTypeError: If the file is not a PDF.
            FileTooLargeError: If the file exceeds MAX_UPLOAD_SIZE_MB.
            ProcessingError: If any processing step fails.
        """
        # ── Validate extension ───────────────────────────────────────
        ext = Path(filename).suffix.lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise InvalidFileTypeError(filename, settings.ALLOWED_EXTENSIONS)

        # ── Validate size ────────────────────────────────────────────
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > settings.MAX_UPLOAD_SIZE_MB:
            raise FileTooLargeError(size_mb, settings.MAX_UPLOAD_SIZE_MB)

        # ── Generate ID and save file ────────────────────────────────
        document_id = str(uuid.uuid4())
        safe_filename = f"{document_id}_{filename}"
        upload_path = Path(settings.UPLOAD_DIR) / safe_filename

        try:
            with open(upload_path, "wb") as f:
                f.write(file_bytes)
            logger.info(f"Saved upload: {upload_path} ({size_mb:.2f} MB)")
        except OSError as exc:
            raise ProcessingError(f"Failed to save file: {exc}") from exc

        # ── Extract text ────────────────────────────────────────────
        try:
            extraction = self._pdf_service.extract_text(str(upload_path))
        except Exception as exc:
            # Clean up on failure
            self._cleanup_file(str(upload_path))
            raise ProcessingError(f"Text extraction failed: {exc}") from exc

        # ── Preprocess pages ─────────────────────────────────────────
        pages = self._pdf_service.preprocess_pages(
            extraction["pages"], extraction.get("metadata")
        )

        # ── Chunk document ───────────────────────────────────────────
        chunks = self._pdf_service.chunk_document(pages)

        if not chunks:
            self._cleanup_file(str(upload_path))
            raise ProcessingError("No text chunks could be extracted from the document")

        # ── Validate chunks ──────────────────────────────────────────
        warnings = self._analytics_service.validate_chunks(chunks)
        if warnings:
            logger.warning(f"Chunk validation warnings for {document_id}: {len(warnings)}")

        # ── Generate embeddings ──────────────────────────────────────
        chunk_texts = [c["text"] for c in chunks]
        try:
            embeddings = self._embedding_generator.generate(chunk_texts)
        except Exception as exc:
            self._cleanup_file(str(upload_path))
            raise ProcessingError(f"Embedding generation failed: {exc}") from exc

        # ── Prepare metadata and IDs for vector store ────────────────
        chunk_ids: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for i, chunk in enumerate(chunks):
            cid = f"{document_id}_chunk_{i:06d}"
            chunk_ids.append(cid)
            metadatas.append({
                "document_id": document_id,
                "filename": filename,
                "chunk_index": i,
                "page_numbers": chunk["page_numbers"],
                "char_count": chunk["char_count"],
            })

        # ── Store in ChromaDB ────────────────────────────────────────
        try:
            self._vector_store.add_chunks(chunk_ids, embeddings, chunk_texts, metadatas)
        except Exception as exc:
            self._cleanup_file(str(upload_path))
            raise ProcessingError(f"Vector store insertion failed: {exc}") from exc

        # ── Generate and register metadata ───────────────────────────
        doc_metadata = self._analytics_service.generate_metadata(
            document_id=document_id,
            filename=filename,
            file_size_bytes=len(file_bytes),
            pdf_metadata=extraction.get("metadata", {}),
            pages=pages,
            chunks=chunks,
        )
        self._analytics_service.register_document(document_id, doc_metadata)

        logger.info(
            f"Document processed successfully | id={document_id} "
            f"| pages={len(pages)} | chunks={len(chunks)}"
        )

        return {
            "document_id": document_id,
            "filename": filename,
            "pages": len(pages),
            "chunks": len(chunks),
            "status": "success",
            "message": "Document processed successfully",
        }

    async def get_all_documents(self) -> list[dict[str, Any]]:
        """Return all processed documents with summary metadata."""
        documents = self._analytics_service.get_all_documents()
        result: list[dict[str, Any]] = []
        for doc in documents:
            result.append({
                "document_id": doc["document_id"],
                "filename": doc["filename"],
                "pages": doc["total_pages"],
                "chunks": doc["total_chunks"],
                "file_size_bytes": doc["file_size_bytes"],
                "created_at": doc["created_at"],
                "status": doc["status"],
            })
        return result

    async def delete_document(self, document_id: str) -> dict[str, Any]:
        """
        Delete a document and all its data from the system.

        Removes:
            - Vector store entries
            - Analytics registry entry
            - Uploaded file
            - Report JSON

        Args:
            document_id: The document to delete.

        Returns:
            Deletion confirmation dict.

        Raises:
            DocumentNotFoundError: If the document doesn't exist.
        """
        doc = self._analytics_service.get_document(document_id)
        if not doc:
            raise DocumentNotFoundError(document_id)

        # Remove from vector store
        deleted_chunks = self._vector_store.delete_document(document_id)

        # Remove from analytics registry
        self._analytics_service.remove_document(document_id)

        # Delete uploaded file
        filename = doc.get("filename", "")
        safe_filename = f"{document_id}_{filename}"
        file_path = Path(settings.UPLOAD_DIR) / safe_filename
        self._cleanup_file(str(file_path))

        # Delete report JSON
        report_path = Path(settings.REPORTS_DIR) / f"{document_id}.json"
        self._cleanup_file(str(report_path))

        logger.info(f"Document deleted: {document_id} ({deleted_chunks} chunks removed)")

        return {
            "document_id": document_id,
            "status": "deleted",
            "message": f"Document '{doc.get('filename', document_id)}' deleted successfully",
        }

    def get_health_stats(self) -> dict[str, Any]:
        """Return health statistics for the system."""
        stats = self._analytics_service.compute_dataset_statistics()
        return {
            "documents_indexed": stats.get("total_documents", 0),
            "total_chunks": stats.get("total_chunks", 0),
        }

    @staticmethod
    def _cleanup_file(file_path: str) -> None:
        """Safely delete a file if it exists."""
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                logger.debug(f"Cleaned up file: {file_path}")
        except OSError as exc:
            logger.warning(f"Failed to clean up file {file_path}: {exc}")