# OmniBrain AI/ML Module

This module contains the AI/ML engineering deliverables for **Project 1: OmniBrain**.

## Responsibilities & Deliverables
- **VLM Integration**: Process PDFs and visual components (tables/charts) using PyMuPDF and VLM (Gemini/OpenAI) to extract structured JSON.
- **Vision Agent**: Orchestrate image extraction and visual reasoning.
- **Self-RAG Pipeline**: A self-correcting retrieval loop using LangGraph that validates document relevance, rewrites search queries upon failure, and performs hallucination checks.

## Project Structure
- `config.py`: Key loading and model config.
- `vision_vlm.py`: PyMuPDF rendering and structured VLM JSON parser.
- `retriever.py`: In-memory semantic/keyword retrieval database.
- `self_rag.py`: LangGraph StateGraph state machine.
- `test_vlm.py`: Test runner that programmatically draws a PDF and executes the VLM.
- `run.py`: Full demo script showcasing standard RAG, Self-RAG query rewriting, and the Vision Agent.
- `omnibrain_colab.ipynb`: Fully compiled Jupyter Notebook for Google Colab and VS Code.

## Quick Start
```bash
# Install dependencies and run tests
uv run python test_vlm.py

# Run full integration pipeline
uv run python run.py
```
