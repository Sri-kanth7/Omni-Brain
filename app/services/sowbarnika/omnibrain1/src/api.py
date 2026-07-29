import os
import sys
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# Ensure src is in path
sys.path.insert(0, os.path.dirname(__file__))

from retrieval import MultiModalVectorStore
from orchestrator import OmniBrainOrchestrator
from guardrails import GuardrailsManager
from evaluation import OmniBrainEvaluator
from ingestion import ingest_pdf
from db_setup import setup_database

# Initialize database
setup_database()

app = FastAPI(title="OmniBrain API Backend", version="1.0.0")

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder for serving extracted charts/images
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
static_path = os.path.join(PROJECT_ROOT, "static")
os.makedirs(static_path, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_path), name="static")

# Shared state
vector_store = MultiModalVectorStore()
orchestrator = OmniBrainOrchestrator(vector_store)
guardrails = GuardrailsManager()
evaluator = OmniBrainEvaluator()

class QueryRequest(BaseModel):
    query: str
    image_path: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    intermediate_steps: List[str]
    citations: List[Dict[str, Any]]
    evaluation: Dict[str, Any]
    sql_result: Optional[Dict[str, Any]] = None
    vision_result: Optional[Dict[str, Any]] = None
    search_result: Optional[Dict[str, Any]] = None
    image_path: Optional[str] = None
    allowed: bool
    refusal_reason: Optional[str] = None

def run_background_ingestion(file_path: str):
    """Background task to ingest PDF files."""
    try:
        res = ingest_pdf(file_path, vector_store)
        print(f"Background ingestion completed: {res}")
    except Exception as e:
        print(f"Background ingestion failed: {e}")

@app.post("/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Uploads a PDF report and triggers asynchronous parsing and ingestion."""
    upload_dir = os.path.join(PROJECT_ROOT, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    # Trigger ingestion asynchronously
    background_tasks.add_task(run_background_ingestion, file_path)
    
    return {
        "filename": file.filename,
        "status": "processing",
        "message": "Ingestion triggered asynchronously in the background."
    }

@app.post("/query", response_model=QueryResponse)
async def query_omnibrain(req: QueryRequest):
    """Processes user query through input rails, LangGraph orchestrator, output rails, and evaluation."""
    # 1. Run Input Guardrails
    input_check = guardrails.validate_input(req.query)
    if not input_check["allowed"]:
        return QueryResponse(
            query=req.query,
            answer=input_check["refusal"],
            intermediate_steps=[],
            citations=[],
            evaluation={"groundedness": 0.0, "relevance": 0.0, "hallucination_score": 1.0, "status": "blocked"},
            allowed=False,
            refusal_reason=input_check["refusal"]
        )
        
    # 2. Run LangGraph Orchestrator
    # Check if there is an image to search for or reference
    image_path = req.image_path
    if not image_path and vector_store.images:
        # Fallback to the latest ingested image
        image_path = vector_store.images[-1]["image_path"]
        # Convert path if it has static prefix
        if image_path.startswith("static/"):
            image_path = image_path[len("static/"):]
            
    res = orchestrator.run(req.query, image_path=image_path)
    
    # 3. Run Output Guardrails
    output_check = guardrails.validate_output(res["final_answer"])
    final_answer = output_check["replacement"]
    
    # 4. Run Offline Evaluation Pipeline
    eval_metrics = evaluator.evaluate(req.query, final_answer, res["citations"])
    
    return QueryResponse(
        query=req.query,
        answer=final_answer,
        intermediate_steps=res.get("intermediate_steps", []),
        citations=res.get("citations", []),
        evaluation=eval_metrics,
        sql_result=res.get("sql_result"),
        vision_result=res.get("vision_result"),
        search_result=res.get("search_result"),
        image_path=res.get("image_path"),
        allowed=True
    )

@app.get("/status")
async def get_status():
    """Returns database sizes and active configuration."""
    return {
        "text_chunks": len(vector_store.texts),
        "image_chunks": len(vector_store.images),
        "environment": "mock" if vector_store.is_mock else "live",
        "openai_available": bool(os.getenv("OPENAI_API_KEY"))
    }
