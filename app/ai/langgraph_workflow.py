"""
LangGraph workflow definition for the OmniBrain RAG pipeline.

Responsibilities:
- Validate workflow input
- Orchestrate Gemini generation
- Keep workflow logic independent of fallback handling

NOTE:
Fallback behaviour belongs in ChatService, where retrieved chunks and
source metadata are available.
"""

from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.ai.gemini_service import GeminiService
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class RAGState(TypedDict):
    question: str
    context: str
    temperature: Optional[float]
    answer: str
    model_used: str
    error: Optional[str]


def validate_input(state: RAGState) -> RAGState:
    """Validate the incoming request."""

    logger.debug("Node: validate_input")

    question = state.get("question", "").strip()
    context = state.get("context", "").strip()

    if not question:
        state["error"] = "Question cannot be empty."
        state["answer"] = "Please provide a valid question."
        return state

    if not context:
        state["error"] = "No context available."
        state["answer"] = (
            "I couldn't find any relevant information in the uploaded documents."
        )
        return state

    state["error"] = None
    return state


def generate_answer(state: RAGState) -> RAGState:
    """
    Generate the answer using Gemini.

    Any Gemini exception intentionally propagates back to ChatService.
    ChatService is responsible for graceful fallback using retrieved chunks.
    """

    logger.debug("Node: generate_answer")

    if state.get("error"):
        return state

    logger.info(
        "Generating Gemini response | Question=%d chars | Context=%d chars",
        len(state["question"]),
        len(state["context"]),
    )

    gemini = GeminiService()

    answer = gemini.generate_response(
        question=state["question"],
        context=state["context"],
        temperature=state.get("temperature"),
    )

    answer = (answer or "").strip()

    if not answer:
        raise RuntimeError("Gemini returned an empty response.")

    state["answer"] = answer
    state["model_used"] = settings.GEMINI_MODEL
    state["error"] = None

    return state


def format_output(state: RAGState) -> RAGState:
    """Normalize workflow output."""

    logger.debug("Node: format_output")

    state["answer"] = state.get("answer", "").strip()

    if not state.get("model_used"):
        state["model_used"] = settings.GEMINI_MODEL

    return state


def has_error(state: RAGState) -> str:
    return "skip" if state.get("error") else "proceed"


class LangGraphWorkflow:
    """
    LangGraph orchestration for OmniBrain.

        validate_input
              |
              v
       generate_answer
              |
              v
        format_output
              |
              v
             END
    """

    def __init__(self) -> None:
        self._graph = self._build_graph()
        logger.info("LangGraphWorkflow initialized.")

    def _build_graph(self):
        workflow = StateGraph(RAGState)

        workflow.add_node("validate_input", validate_input)
        workflow.add_node("generate_answer", generate_answer)
        workflow.add_node("format_output", format_output)

        workflow.set_entry_point("validate_input")

        workflow.add_conditional_edges(
            "validate_input",
            has_error,
            {
                "proceed": "generate_answer",
                "skip": "format_output",
            },
        )

        workflow.add_edge("generate_answer", "format_output")
        workflow.add_edge("format_output", END)

        return workflow.compile()

    def run(
        self,
        question: str,
        context: str,
        temperature: Optional[float] = None,
    ) -> dict[str, Any]:

        logger.info("Running LangGraph workflow.")

        state: RAGState = {
            "question": question,
            "context": context,
            "temperature": temperature,
            "answer": "",
            "model_used": "",
            "error": None,
        }

        result: RAGState = self._graph.invoke(state)

        logger.info(
            "Workflow completed successfully | Answer Length=%d",
            len(result.get("answer", "")),
        )

        return {
            "answer": result.get("answer", ""),
            "model": result.get("model_used", settings.GEMINI_MODEL),
        }
