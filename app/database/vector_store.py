"""
ChromaDB integration for storing and retrieving document embeddings.
Provides a high-level interface for vector search operations.
"""

from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """
    Wrapper around ChromaDB for document embedding storage and retrieval.

    Manages a single persistent collection where each document chunk is stored
    with its embedding, metadata, and source tracking.
    """

    def __init__(self) -> None:
        """Initialize the ChromaDB client and collection."""
        self._client: chromadb.PersistentClient = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection: chromadb.Collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"VectorStore initialized | collection='{settings.CHROMA_COLLECTION_NAME}' "
            f"| persist_dir='{settings.CHROMA_PERSIST_DIR}'"
        )

    # ── Write Operations ─────────────────────────────────────────────

    def add_chunks(
        self,
        chunk_ids: list[str],
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> int:
        """
        Add document chunks to the vector store.

        Args:
            chunk_ids: Unique IDs for each chunk.
            embeddings: Vector embeddings for each chunk.
            texts: Original text content of each chunk.
            metadatas: Metadata dicts (document_id, page, chunk_index, etc.).

        Returns:
            Number of chunks added.
        """
        if not chunk_ids:
            logger.warning("add_chunks called with empty chunk_ids")
            return 0

        self._collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info(f"Added {len(chunk_ids)} chunks to vector store")
        return len(chunk_ids)

    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks belonging to a document.

        Args:
            document_id: The document ID to remove.

        Returns:
            Number of chunks deleted.
        """
        results = self._collection.get(where={"document_id": document_id})
        chunk_ids = results.get("ids", [])
        if chunk_ids:
            self._collection.delete(ids=chunk_ids)
            logger.info(f"Deleted {len(chunk_ids)} chunks for document_id='{document_id}'")
        else:
            logger.warning(f"No chunks found for document_id='{document_id}'")
        return len(chunk_ids)

    def clear_all(self) -> int:
        """
        Delete all chunks from the collection.

        Returns:
            Number of chunks deleted.
        """
        count = self._collection.count()
        if count > 0:
            all_ids = self._collection.get()["ids"]
            self._collection.delete(ids=all_ids)
            logger.info(f"Cleared all {count} chunks from vector store")
        return count

    # ── Read Operations ──────────────────────────────────────────────

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Search the vector store for the most similar chunks.

        Args:
            query_embedding: The embedding vector of the query.
            top_k: Number of top results to return.
            where: Optional filter (e.g., {"document_id": "..."}).

        Returns:
            List of result dicts with keys: id, document, metadata, distance.
        """
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        formatted: list[dict[str, Any]] = []
        if results["ids"] and results["ids"][0]:
            for idx, chunk_id in enumerate(results["ids"][0]):
                formatted.append({
                    "id": chunk_id,
                    "document": results["documents"][0][idx],
                    "metadata": results["metadatas"][0][idx],
                    "distance": float(results["distances"][0][idx]),
                    "similarity": 1.0 - float(results["distances"][0][idx]),
                })

        logger.debug(f"Search returned {len(formatted)} results (top_k={top_k})")
        return formatted

    # ── Stats ────────────────────────────────────────────────────────

    def count(self) -> int:
        """Return total number of chunks in the collection."""
        return self._collection.count()

    def get_document_ids(self) -> list[str]:
        """Return unique document IDs present in the store."""
        all_metadatas = self._collection.get(include=["metadatas"])["metadatas"]
        return list({m["document_id"] for m in all_metadatas if m})

    def get_document_chunk_count(self, document_id: str) -> int:
        """Return the number of chunks for a given document."""
        results = self._collection.get(where={"document_id": document_id})
        return len(results.get("ids", []))