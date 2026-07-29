# OmniBrain: Agentic Multi-Modal RAG Orchestrator

OmniBrain is an agentic multi-modal RAG orchestrator designed for grounded financial analysis. It handles complex queries over multi-modal financial documents and databases by routing tasks dynamically using a LangGraph supervisor agent, retrieving data from a multi-modal vector database (text + chart/table images), querying historical stock data via a Text-to-SQL agent, and parsing charts/tables using a Vision-Language Model (VLM).

To facilitate local development and automated testing offline, all systems fall back gracefully to a robust dry-run/mock mode when API keys are missing or when `OMNIBRAIN_ENV=mock` is configured.

---

## Codebase Architecture

```
d:\OmniBrain\
├── config/
│   ├── config.yml           # NeMo Guardrails configuration
│   └── rails.co             # NeMo Guardrails policy flows
├── src/
│   ├── __init__.py
│   ├── config.py            # Environment configuration and fallback detection
│   ├── db_setup.py          # SQLite database schema setup and data populator
│   ├── retrieval.py         # Multi-modal vector store (Text & Images) via cosine similarity
│   ├── guardrails.py        # Input/Output validation & NeMo Guardrails wrapper
│   ├── monitoring.py        # Langfuse telemetry CallbackHandler setup
│   ├── evaluation.py        # Groundedness, relevance, and hallucination scoring pipeline
│   ├── orchestrator.py      # LangGraph supervisor StateGraph definition
│   ├── main.py              # End-to-end orchestration and guardrail demonstration
│   └── agents/
│       ├── __init__.py
│       ├── search_agent.py  # Semantic text/image retrieval agent
│       ├── sql_agent.py     # Text-to-SQL pricing database query agent
│       └── vision_agent.py  # Visual balance sheet / chart parser agent
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Pytest fixtures and mock setups
│   ├── test_agents.py       # Unit tests for individual agents
│   ├── test_retrieval.py    # Unit tests for MultiModalVectorStore
│   ├── test_orchestrator.py # Integration tests for LangGraph state machine flow
│   └── test_guardrails.py   # Unit tests for input/output guardrail checks
├── requirements.txt         # Dependencies
├── .env.example             # Env variable template
└── README.md                # System documentation
```

---

## Setup & Installation

1. **Clone/Move into the workspace**:
   ```bash
   cd d:\OmniBrain
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and fill in the API credentials. To run in fully mock mode (for offline testing), set `OMNIBRAIN_ENV=mock`:
   ```bash
   OMNIBRAIN_ENV=mock
   ```

4. **Initialize Database**:
   Set up the SQLite stock data database:
   ```bash
   python src/db_setup.py
   ```

---

## Running Verification

### 1. Automated Test Suite
To run all unit and integration tests under pytest:
```bash
$env:PYTHONPATH="d:\OmniBrain"
python -m pytest tests -v
```

### 2. End-to-End Orchestrator Demonstration
Run the demonstration script to trace a sample multi-modal investment memo query, check input/output guardrails compliance, and view evaluation pipeline scoring outputs:
```bash
python src/main.py
```
