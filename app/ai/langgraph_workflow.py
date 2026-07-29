"""
LangGraph workflow definition for the OmniBrain RAG pipeline.

Gowtham's Module:
- Defines a state-graph pipeline using LangGraph
- Orchestrates: retrieval → context building → Gemini generation → output formatting
- Provides a clean .run() interface for the ChatService
"""

from typing import Any, Optional, TypedDict

from langgraph.graph import StateGraph, END

from app.ai.gemini_service import GeminiService
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Graph State Definition
# ═══════════════════════════════════════════════════════════════════

class RAGState(TypedDict):
    """
    Represents the state of the RAG pipeline at any node.

    Attributes:
        question: The user's original question.
        context: Retrieved document context as a formatted string.
        temperature: Optional temperature override.
        answer: The generated answer.
        model_used: The model identifier used for generation.
        error: Error message if a node fails.
    """
    question: str
    context: str
    temperature: Optional[float]
    answer: str
    model_used: str
    error: Optional[str]


# ═══════════════════════════════════════════════════════════════════
# Node Functions
# ═══════════════════════════════════════════════════════════════════

def validate_input(state: RAGState) -> RAGState:
    """
    Validate the input state before processing.

    Checks:
    - question is non-empty
    - context is non-empty (if available)

    Returns the state with error set if validation fails.
    """
    logger.debug("Node: validate_input")

    if not state.get("question", "").strip():
        state["error"] = "Question cannot be empty"
        state["answer"] = "Please provide a valid question."
        return state

    if not state.get("context", "").strip():
        state["error"] = "No context provided"
        state["answer"] = "I don't have any document context to answer from."
        return state

    state["error"] = None
    return state


def generate_answer(state: RAGState) -> RAGState:
    """
    Generate an answer using the Gemini service.

    Takes the question and context from state and calls the LLM.
    On success, populates 'answer' and 'model_used'.
    """
    logger.debug("Node: generate_answer")

    if state.get("error"):
        return state

    try:
        gemini = GeminiService()
        answer = gemini.generate_response(
            question=state["question"],
            context=state["context"],
            temperature=state.get("temperature"),
        )
        state["answer"] = answer
        state["model_used"] = settings.GEMINI_MODEL
        state["error"] = None

    except Exception as exc:
        logger.error(f"Answer generation failed: {exc}")
        state["answer"] = "I encountered an error while generating the response. Please try again."
        state["error"] = str(exc)
        state["model_used"] = settings.GEMINI_MODEL

    return state


def format_output(state: RAGState) -> RAGState:
    """
    Final formatting/cleanup of the output.
    Trims the answer and ensures consistent structure.
    """
    logger.debug("Node: format_output")

    if state.get("answer"):
        state["answer"] = state["answer"].strip()

    if not state.get("model_used"):
        state["model_used"] = settings.GEMINI_MODEL

    return state


# ═══════════════════════════════════════════════════════════════════
# Conditional Edge Logic
# ═══════════════════════════════════════════════════════════════════

def has_error(state: RAGState) -> str:
    """
    Route decision: if there's an error, skip to format_output;
    otherwise proceed to generate_answer.
    """
    if state.get("error"):
        logger.debug("Routing: error detected → format_output")
        return "skip"
    return "proceed"


# ═══════════════════════════════════════════════════════════════════
# Graph Construction
# ═══════════════════════════════════════════════════════════════════

class LangGraphWorkflow:
    """
    LangGraph-based RAG workflow for OmniBrain.

    Graph structure:
        validate_input → (has_error?) → generate_answer → format_output → END
                            │                │
                            └── skip ────────┘

    Usage:
        workflow = LangGraphWorkflow()
        result = workflow.run(question="...", context="...")
    """

    def __init__(self) -> None:
        """Build and compile the LangGraph state graph."""
        self._graph = self._build_graph()
        logger.info("LangGraphWorkflow initialized")

    def _build_graph(self) -> StateGraph:
        """Construct the state graph with nodes and edges."""
        workflow = StateGraph(RAGState)

        # ── Add nodes ───────────────────────────────────────────────
        workflow.add_node("validate_input", validate_input)
        workflow.add_node("generate_answer", generate_answer)
        workflow.add_node("format_output", format_output)

        # ── Set entry point ─────────────────────────────────────────
        workflow.set_entry_point("validate_input")

        # ── Add edges ───────────────────────────────────────────────
        # validate_input → conditional → generate_answer or format_output
        workflow.add_conditional_edges(
            "validate_input",
            has_error,
            {
                "proceed": "generate_answer",
                "skip": "format_output",
            },
        )

        # generate_answer → format_output
        workflow.add_edge("generate_answer", "format_output")

        # format_output → END
        workflow.add_edge("format_output", END)

        return workflow.compile()

    def run(
        self,
        question: str,
        context: str,
        temperature: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Execute the RAG workflow.

        Args:
            question: The user's question.
            context: Retrieved document context.
            temperature: Optional model temperature override.

        Returns:
            Dict with 'answer' and 'model' keys.
        """
        initial_state: RAGState = {
            "question": question,
            "context": context,
            "temperature": temperature,
            "answer": "",
            "model_used": "",
            "error": None,
        }

        logger.info("Running LangGraph workflow")
        final_state: RAGState = self._graph.invoke(initial_state)

        logger.info(f"Workflow complete | answer_len={len(final_state.get('answer', ''))}")
        return {
            "answer": final_state.get("answer", ""),
            "model": final_state.get("model_used", settings.GEMINI_MODEL),
        }