"""
Google Gemini API integration service.

Gowtham's Module:
- Gemini API client management
- Prompt engineering and templating
- Chat response generation via Gemini models
"""

from typing import Any, Optional

import google.generativeai as genai

from app.core.config import settings
from app.core.exceptions import GeminiAPIError
from app.core.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# System prompt template for the RAG Q&A assistant
# ═══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are OmniBrain, an expert AI research assistant. Your role is to answer questions based strictly on the provided document context.

## Instructions

1. **Answer ONLY from the context provided below.** Do not use your general knowledge unless the context is empty.
2. If the context does not contain enough information to answer the question, say: "I cannot find enough information in the provided documents to answer this question."
3. **Cite your sources** by referencing the source number in square brackets, e.g., [Source 1], [Source 2].
4. If multiple sources support the same point, cite all relevant ones.
5. Provide detailed, thorough answers. Include relevant quotes from the context.
6. If the question is ambiguous, ask for clarification.
7. Maintain a professional, helpful tone.

## Context

{context}

## Question

{question}"""


class GeminiService:
    """
    Wraps the Google Generative AI (Gemini) API.

    Provides methods for:
    - Generating text responses from prompts
    - Configurable model parameters (temperature, max tokens)
    - Error handling and retries
    """

    def __init__(self) -> None:
        """Initialize the Gemini client with the configured API key."""
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            logger.warning("GEMINI_API_KEY is not set. Gemini calls will fail at runtime.")

        genai.configure(api_key=api_key)
        self._model_name = settings.GEMINI_MODEL
        self._temperature = settings.GEMINI_TEMPERATURE
        self._max_tokens = settings.GEMINI_MAX_TOKENS

        logger.info(f"GeminiService initialized | model={self._model_name}")

    def generate_response(
        self,
        question: str,
        context: str,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate an answer from Gemini given a question and retrieved context.

        Args:
            question: The user's question.
            context: Retrieved document chunks as a formatted string.
            temperature: Override temperature (None = use default).

        Returns:
            The generated answer text.

        Raises:
            GeminiAPIError: If the API call fails.
        """
        prompt = SYSTEM_PROMPT.format(question=question, context=context)
        temp = temperature if temperature is not None else self._temperature

        logger.debug(f"Generating response | temp={temp} | context_len={len(context)}")

        try:
            model = genai.GenerativeModel(
                model_name=self._model_name,
                generation_config={
                    "temperature": temp,
                    "max_output_tokens": self._max_tokens,
                    "top_p": 0.95,
                    "top_k": 40,
                },
            )

            response = model.generate_content(prompt)
            answer = response.text.strip()

            logger.info(f"Response generated | {len(answer)} chars")
            return answer

        except Exception as exc:
            error_msg = f"Gemini API call failed: {exc}"
            logger.error(error_msg)
            raise GeminiAPIError(error_msg) from exc

    def generate_chat_response(
        self,
        question: str,
        context: str,
        history: Optional[list[dict[str, str]]] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a chat-style response with optional conversation history.

        Args:
            question: The user's question.
            context: Retrieved document context.
            history: Optional list of {"role": "user"/"model", "parts": [...]} dicts.
            temperature: Override temperature.

        Returns:
            The generated answer text.
        """
        prompt = SYSTEM_PROMPT.format(question=question, context=context)
        temp = temperature if temperature is not None else self._temperature

        try:
            model = genai.GenerativeModel(
                model_name=self._model_name,
                generation_config={
                    "temperature": temp,
                    "max_output_tokens": self._max_tokens,
                },
            )

            chat = model.start_chat(history=history or [])
            response = chat.send_message(prompt)
            return response.text.strip()

        except Exception as exc:
            error_msg = f"Gemini chat API call failed: {exc}"
            logger.error(error_msg)
            raise GeminiAPIError(error_msg) from exc

    def get_model_info(self) -> dict[str, Any]:
        """Return information about the configured Gemini model."""
        return {
            "model": self._model_name,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }