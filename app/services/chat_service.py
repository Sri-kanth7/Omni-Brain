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
    workflow for answer generation.

    If answer generation is unavailable for any reason, the service falls
    back to presenting the retrieved document excerpts directly, so the
    user still receives a useful, well-formatted response.
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
        """
        start_time = time.time()
        logger.info(
            f"Processing question (top_k={top_k}, docs_filter={document_ids is not None})"
        )

        # ── 1. Generate query embedding ──────────────────────────────
        try:
            query_embedding = self._embedding_generator.generate_single(question)
        except Exception as exc:
            raise ProcessingError(f"Failed to generate query embedding: {exc}") from exc

        # ── 2. Search vector store ───────────────────────────────────
        where_filter = None
        if document_ids:
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
            elapsed = int((time.time() - start_time) * 1000)
            return {
                "answer": (
                    "I couldn't find any relevant information in the uploaded "
                    "documents to answer your question."
                ),
                "sources": [],
                "processing_time_ms": elapsed,
                "model_used": settings.GEMINI_MODEL,
            }

        # ── 3. Adaptive filtering ────────────────────────────────────
        filtered_results = self._filter_by_adaptive_threshold(search_results)

        # ── 4. Deduplicate ───────────────────────────────────────────
        filtered_results = self._deduplicate_chunks(filtered_results)

        # ── 5. Build context and sources ─────────────────────────────
        context, sources = self._build_context_and_sources(filtered_results)

        logger.info(
            f"Passing {len(filtered_results)} chunks to workflow "
            f"(context length={len(context)} chars)"
        )

        # ── 6. Generate answer ───────────────────────────────────────
        answer, model_used = self._generate_answer(
            question=question,
            context=context,
            temperature=temperature,
            filtered_results=filtered_results,
        )

        elapsed = int((time.time() - start_time) * 1000)

        logger.info(
            f"Question answered in {elapsed}ms "
            f"| Sources used: {len(sources)}"
        )

        return {
            "answer": answer,
            "sources": sources,
            "processing_time_ms": elapsed,
            "model_used": model_used,
        }

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _filter_by_adaptive_threshold(
        search_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Filter search results using a similarity threshold that adapts to
        how strong the best match is, guaranteeing a minimum amount of
        context even when nothing clears the threshold.
        """
        best_similarity = max(r["similarity"] for r in search_results)

        if best_similarity >= 0.80:
            dynamic_threshold = best_similarity - 0.10
        elif best_similarity >= 0.60:
            dynamic_threshold = best_similarity - 0.15
        else:
            dynamic_threshold = best_similarity - 0.20

        # Never let the threshold become too low
        dynamic_threshold = max(dynamic_threshold, 0.20)

        logger.info(
            f"Best similarity={best_similarity:.4f}, "
            f"Dynamic threshold={dynamic_threshold:.4f}"
        )

        filtered_results = []
        for result in search_results:
            logger.info(
                f"Similarity={result['similarity']:.4f}, "
                f"Distance={result['distance']:.4f}"
            )
            if result["similarity"] >= dynamic_threshold:
                filtered_results.append(result)

        # Guarantee minimum context
        if len(filtered_results) < 2:
            logger.info(
                "Too few chunks after filtering. "
                "Using top search results instead."
            )
            filtered_results = search_results[: min(2, len(search_results))]

        return filtered_results

    @staticmethod
    def _deduplicate_chunks(
        filtered_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Remove chunks with duplicate text content, keeping the first
        (highest-ranked) occurrence of each unique chunk.
        """
        seen_texts: set[str] = set()
        deduplicated: list[dict[str, Any]] = []

        for result in filtered_results:
            normalized_text = result["document"].strip()
            if normalized_text in seen_texts:
                continue
            seen_texts.add(normalized_text)
            deduplicated.append(result)

        removed = len(filtered_results) - len(deduplicated)
        if removed:
            logger.info(f"Removed {removed} duplicate chunk(s)")

        return deduplicated

    @staticmethod
    def _build_context_and_sources(
        filtered_results: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Build the LLM context string and the source metadata list from the
        filtered/deduplicated chunks.
        """
        context_parts: list[str] = []
        sources: list[dict[str, Any]] = []

        for i, result in enumerate(filtered_results):
            metadata = result["metadata"]
            similarity = result["similarity"]

            context_parts.append(f"[Source {i + 1}]\n{result['document']}")

            page_numbers = metadata.get("page_numbers", "")

            if isinstance(page_numbers, str):
                try:
                    page_number = int(page_numbers.split(",")[0])
                except Exception:
                    page_number = 1
            elif isinstance(page_numbers, list):
                page_number = page_numbers[0] if page_numbers else 1
            else:
                page_number = 1

            sources.append(
                {
                    "document_id": metadata.get("document_id", ""),
                    "filename": metadata.get("filename", "unknown"),
                    "page_number": page_number,
                    "chunk_index": metadata.get("chunk_index", 0),
                    "similarity_score": round(similarity, 4),
                    "snippet": result["document"],
                }
            )

        context = "\n\n".join(context_parts)
        return context, sources

    def _generate_answer(
        self,
        question: str,
        context: str,
        temperature: Optional[float],
        filtered_results: list[dict[str, Any]],
    ) -> tuple[str, str]:
        """
        Generate the final answer text and the model/source label to report.

        Attempts to generate an answer via the LangGraph workflow. If that
        is unavailable for any reason, falls back to presenting the
        retrieved document excerpts directly as the answer.
        """
        try:
            llm_response = self._langgraph_workflow.run(
                question=question,
                context=context,
                temperature=temperature,
            )
            answer = llm_response.get("answer", "").strip()
            if answer:
                return answer, llm_response.get("model", settings.GEMINI_MODEL)
        except Exception as exc:
            logger.warning(f"Falling back to retrieved excerpts: {exc}")

        return self._build_fallback_answer(filtered_results), settings.GEMINI_MODEL

    @staticmethod
    def _build_fallback_answer(filtered_results: list[dict[str, Any]]) -> str:
        """
        Construct a readable answer directly from the retrieved chunks,
        used when answer generation isn't available.
        """
        sections: list[str] = [
            "Here's what I found in the uploaded documents that's relevant "
            "to your question:\n"
        ]

        for i, result in enumerate(filtered_results):
            metadata = result["metadata"]
            filename = metadata.get("filename", "unknown")
            text = result["document"].strip()

            sections.append(f"**From {filename}:**\n{text}")

        return "\n\n".join(sections)
