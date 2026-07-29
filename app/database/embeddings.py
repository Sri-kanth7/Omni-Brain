"""
Embedding generation using Sentence-Transformers.
Provides a consistent interface for creating and managing embeddings.
"""

from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class EmbeddingGenerator:
    """
    Generates vector embeddings from text using Sentence-Transformers.

    Uses the model specified in settings.EMBEDDING_MODEL (default: all-MiniLM-L6-v2).
    The model is loaded lazily on first use to avoid cold-start delays.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        """
        Initialize the embedding generator.

        Args:
            model_name: Override the default embedding model name.
        """
        self._model_name = model_name or settings.EMBEDDING_MODEL
        self._model: Optional[SentenceTransformer] = None
        self._dimension: int = settings.EMBEDDING_DIMENSION
        logger.info(f"EmbeddingGenerator initialized with model: {self._model_name}")

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load and return the SentenceTransformer model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self._dimension}")
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        if self._model is None:
            return self._dimension
        return self._model.get_sentence_embedding_dimension()

    def generate(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of text strings.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each vector is a list of floats).
        """
        if not texts:
            logger.warning("generate() called with empty text list")
            return []

        logger.debug(f"Generating embeddings for {len(texts)} texts")
        embeddings: np.ndarray = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def generate_single(self, text: str) -> list[float]:
        """
        Generate an embedding for a single text string.

        Args:
            text: The text string to embed.

        Returns:
            A single embedding vector.
        """
        return self.generate([text])[0]