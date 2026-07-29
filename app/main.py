"""
OmniBrain — FastAPI Application Entry Point.

Initializes and configures the FastAPI application with:
- CORS middleware for React frontend connectivity
- Static file serving for uploaded PDFs
- All API routes mounted under /api/v1
- Global exception handlers
- Startup/shutdown lifecycle events
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import settings
from app.core.exceptions import OmniBrainError
from app.core.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan: startup and shutdown events.

    Startup: Log that the server is ready and verify the upload directory exists.
    Shutdown: Log the shutdown event.
    """
    # ── Startup ─────────────────────────────────────────────────────
    logger.info(
        f"{settings.APP_NAME} v{settings.APP_VERSION} starting up",
        extra={
            "host": settings.HOST,
            "port": settings.PORT,
            "debug": settings.DEBUG,
            "auth_enabled": settings.AUTH_ENABLED,
            "embedding_model": settings.EMBEDDING_MODEL,
            "gemini_model": settings.GEMINI_MODEL,
        },
    )

    # Ensure directories exist
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.REPORTS_DIR).mkdir(parents=True, exist_ok=True)

    yield

    # ── Shutdown ────────────────────────────────────────────────────
    logger.info(f"{settings.APP_NAME} shutting down")


# ═══════════════════════════════════════════════════════════════════
# Application Factory
# ═══════════════════════════════════════════════════════════════════

app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ═══════════════════════════════════════════════════════════════════
# CORS — Allow React frontend to connect
# ═══════════════════════════════════════════════════════════════════

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════
# Static Files — Serve uploaded PDFs (for frontend preview)
# ═══════════════════════════════════════════════════════════════════

app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# ═══════════════════════════════════════════════════════════════════
# API Routes
# ═══════════════════════════════════════════════════════════════════

app.include_router(router, prefix="/api/v1")

# ═══════════════════════════════════════════════════════════════════
# Global Exception Handler
# ═══════════════════════════════════════════════════════════════════

@app.exception_handler(OmniBrainError)
async def omnibrain_error_handler(request: Request, exc: OmniBrainError) -> JSONResponse:
    """Handle custom OmniBrain exceptions with structured error responses."""
    logger.warning(
        f"OmniBrainError: {exc.message}",
        extra={"status_code": exc.status_code, "details": exc.details},
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.message,
            "status_code": exc.status_code,
            "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unhandled exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected internal error occurred",
            "status_code": 500,
            "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        },
    )


# ═══════════════════════════════════════════════════════════════════
# Root endpoint
# ═══════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Welcome endpoint."""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
    }