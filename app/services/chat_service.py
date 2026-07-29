"""
Chat / Q&A service.
Orchestrates retrieval-augmented generation (RAG) pipeline:

1. Accept user question
2. Generate query embedding
3. Search vector store for relevant chunks
4. Build context with source citations
5. Send to LangGraph workflow (→ Gemini)
6. Return answer with citations
"""

import time
from typing import Any, Optional

from app.core.config import settings
from app.core.exceptions import ProcessingError
from app.core.logger import get_logger
from app.database.embeddings import EmbeddingGenerator
from app.database.vector_store import VectorStore
from app.ai.langgraph_workflow import LangGraphWorkflow

logger = get_logger(__name__)


class ChatService:
    """
    Retrieval-Augmented Generation (RAG) service.

    Embeds the user's query, retrieves the most relevant document chunks
    from the vector store, and passes them as context to the LangGraph
    workflow which interfaces with Google Gemini for answer generation.
    """

    def __init__(
        self,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        vector_store: Optional[VectorStore] = None,
        langgraph_workflow: Optional[LangGraphWorkflow] = None,
    ) -> None:
        self._embedding_generator = embedding_generator or EmbeddingGenerator()
        self._vector_store = vector_store or VectorStore()
        self._langgraph_workflow = langgraph_workflow or LangGraphWorkflow()

    async def ask(
        self,
        question: str,
        document_ids: Optional[list[str]] = None,
        top_k: int = 5,
        temperature: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Process a user question and return an AI-generated answer with citations.

        Args:
            question: The user's question.
            document_ids: Optional scope to specific documents.
            top_k: Number of relevant chunks to retrieve.
            temperature: Override model temperature (None = use default).

        Returns:
            Dict with keys:
                - answer (str): AI-generated answer
                - sources (list[dict]): Source citations
                - processing_time_ms (int)
                - model_used (str)

        Raises:
            ProcessingError: If any stage of the pipeline fails.
        """
        start_time = time.time()
        logger.info(f"Processing question (top_k={top_k}, docs_filter={document_ids is not None})")

        # ── 1. Generate query embedding ──────────────────────────────
        try:
            query_embedding = self._embedding_generator.generate_single(question)
        except Exception as exc:
            raise ProcessingError(f"Failed to generate query embedding: {exc}") from exc

        # ── 2. Search vector store ───────────────────────────────────
        where_filter = None
        if document_ids and len(document_ids) > 0:
            # ChromaDB $in operator for filtering by document IDs
            where_filter = {"document_id": {"$in": document_ids}}

        try:
            search_results = self._vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                where=where_filter,
            )
        except Exception as exc:
            raise ProcessingError(f"Vector search failed: {exc}") from exc

        if not search_results:
            logger.info("No relevant chunks found for the query")
            elapsed = int((time.time() - start_time) * 1000)
            return {
                "answer": "I couldn't find any relevant information in the uploaded documents to answer your question. Please try uploading more documents or rephrasing your question.",
                "sources": [],
                "processing_time_ms": elapsed,
                "model_used": settings.GEMINI_MODEL,
            }

        # ── 3. Build context with citations ──────────────────────────
        context_parts: list[str] = []
        sources: list[dict[str, Any]] = []

        for i, result in enumerate(search_results):
            metadata = result["metadata"]
            similarity = result["similarity"]

            # Skip low-relevance results
            if similarity < settings.SIMILARITY_THRESHOLD:
                continue

            context_parts.append(f"[Source {i + 1}] {result['document']}")

            sources.append({
                "document_id": metadata.get("document_id", ""),
                "filename": metadata.get("filename", "unknown"),
                "page_number": metadata.get("page_numbers", [1])[0]
                if isinstance(metadata.get("page_numbers"), list)
                else 1,
                "chunk_index": metadata.get("chunk_index", 0),
                "similarity_score": round(similarity, 4),
                "snippet": result["document"][:500],
            })

        context = "\n\n".join(context_parts) if context_parts else ""

        if not context:
            elapsed = int((time.time() - start_time) * 1000)
            return {
                "answer": "I found some potential sources, but none met the relevance threshold. Please try rephrasing your question.",
                "sources": [],
                "processing_time_ms": elapsed,
                "model_used": settings.GEMINI_MODEL,
            }

        # ── 4. Send to LangGraph / Gemini ────────────────────────────
        try:
            llm_response = self._langgraph_workflow.run(
                question=question,
                context=context,
                temperature=temperature,
            )
        except Exception as exc:
            raise ProcessingError(f"AI response generation failed: {exc}") from exc

        elapsed = int((time.time() - start_time) * 1000)
        logger.info(f"Question answered in {elapsed}ms | {len(sources)} sources cited")

        return {
            "answer": llm_response.get("answer", ""),
            "sources": sources,
            "processing_time_ms": elapsed,
            "model_used": llm_response.get("model", settings.GEMINI_MODEL),
        }