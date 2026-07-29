"""
OmniBrain — Streamlit Frontend

A polished, production-ready Streamlit interface for the OmniBrain RAG backend.
Allows users to:
  - Upload PDF documents
  - Ask questions about uploaded documents
  - View AI answers with source citations
  - Manage (list/delete) indexed documents
  - Monitor system health

Usage:
    streamlit run streamlit_app.py
"""

import os
import time
from pathlib import Path
from typing import Any, Optional

import requests
import streamlit as st

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

# Default backend URL — can be overridden via environment variable or sidebar
DEFAULT_API_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="OmniBrain — AI Document Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
# Custom CSS
# ═══════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
    /* ── Global ─────────────────────────────────────────────── */
    .stApp {
        background: #f8f9fc;
    }

    .main-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 24px;
    }

    .main-header h1 {
        font-size: 28px;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    }

    .main-header .badge {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        font-size: 12px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
        letter-spacing: 0.5px;
    }

    /* ── Cards ──────────────────────────────────────────────── */
    .card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06), 0 1px 2px rgba(0, 0, 0, 0.04);
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }

    .card h3 {
        font-size: 16px;
        font-weight: 600;
        color: #1e293b;
        margin-top: 0;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ── Stats ──────────────────────────────────────────────── */
    .stat-grid {
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
    }

    .stat-box {
        background: linear-gradient(135deg, #f0f4ff, #e8eeff);
        border-radius: 12px;
        padding: 16px 20px;
        flex: 1;
        min-width: 140px;
        text-align: center;
        border: 1px solid #dde4ff;
    }

    .stat-box .value {
        font-size: 28px;
        font-weight: 700;
        color: #4f46e5;
    }

    .stat-box .label {
        font-size: 13px;
        color: #64748b;
        font-weight: 500;
        margin-top: 4px;
    }

    /* ── Source citations ────────────────────────────────────── */
    .source-card {
        background: #f8fafc;
        border-radius: 10px;
        padding: 14px 16px;
        border-left: 4px solid #6366f1;
        margin-bottom: 10px;
        font-size: 13px;
    }

    .source-card .filename {
        font-weight: 600;
        color: #1e293b;
    }

    .source-card .meta {
        color: #64748b;
        font-size: 12px;
    }

    .source-card .snippet {
        color: #475569;
        margin-top: 6px;
        font-style: italic;
    }

    .source-card .score {
        display: inline-block;
        background: #e0e7ff;
        color: #4338ca;
        font-size: 11px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 10px;
    }

    /* ── Answer box ──────────────────────────────────────────── */
    .answer-box {
        background: white;
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
        margin-bottom: 16px;
        line-height: 1.7;
    }

    .answer-box h4 {
        color: #1e293b;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
    }

    /* ── Upload area ─────────────────────────────────────────── */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #c7d2fe !important;
        border-radius: 12px !important;
        background: #f8faff !important;
    }

    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #818cf8 !important;
        background: #f0f4ff !important;
    }

    /* ── Buttons ─────────────────────────────────────────────── */
    .stButton button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }

    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
    }

    .stButton button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }

    /* ── Sidebar ─────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid #e2e8f0;
    }

    [data-testid="stSidebar"] .sidebar-content {
        padding: 20px 16px;
    }

    .sidebar-header {
        font-size: 18px;
        font-weight: 700;
        color: #1e293b;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 24px;
    }

    /* ── Status indicators ────────────────────────────────────── */
    .status-ok {
        color: #10b981;
        font-weight: 600;
    }

    .status-error {
        color: #ef4444;
        font-weight: 600;
    }

    /* ── Delete button ────────────────────────────────────────── */
    .delete-btn button {
        background: #fee2e2 !important;
        color: #dc2626 !important;
        border: 1px solid #fecaca !important;
        font-size: 12px !important;
        padding: 2px 12px !important;
    }

    .delete-btn button:hover {
        background: #fecaca !important;
    }

    /* ── Responsive ───────────────────────────────────────────── */
    @media (max-width: 768px) {
        .stat-grid {
            flex-direction: column;
        }
    }
</style>
"""


# ═══════════════════════════════════════════════════════════════════
# API Client
# ═══════════════════════════════════════════════════════════════════

class OmniBrainAPI:
    """Lightweight API client for the OmniBrain backend."""

    def __init__(self, base_url: str, api_key: Optional[str] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._headers: dict[str, str] = {}
        if api_key:
            self._headers["X-API-Key"] = api_key

    def _url(self, path: str) -> str:
        return f"{self.base_url}{API_PREFIX}{path}"

    def _request(
        self, method: str, path: str, **kwargs: Any
    ) -> requests.Response:
        headers = {**self._headers, **kwargs.pop("headers", {})}
        resp = requests.request(
            method, self._url(path), headers=headers, timeout=120, **kwargs
        )
        return resp

    # ── Health ──────────────────────────────────────────────────────

    def health(self) -> Optional[dict[str, Any]]:
        try:
            resp = self._request("GET", "/health")
            if resp.status_code == 200:
                return resp.json()
            return None
        except requests.RequestException:
            return None

    # ── Upload ──────────────────────────────────────────────────────

    def upload(self, file_path: str, filename: str) -> Optional[dict[str, Any]]:
        try:
            with open(file_path, "rb") as f:
                resp = self._request(
                    "POST",
                    "/upload",
                    files={"file": (filename, f, "application/pdf")},
                )
            if resp.status_code == 200:
                return resp.json()
            return {"error": resp.json().get("detail", resp.text)}
        except requests.RequestException as exc:
            return {"error": str(exc)}

    # ── Chat ────────────────────────────────────────────────────────

    def chat(
        self,
        question: str,
        document_ids: Optional[list[str]] = None,
        top_k: int = 5,
        temperature: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        try:
            payload: dict[str, Any] = {
                "question": question,
                "top_k": top_k,
            }
            if document_ids:
                payload["document_ids"] = document_ids
            if temperature is not None:
                payload["temperature"] = temperature

            resp = self._request("POST", "/chat", json=payload)
            if resp.status_code == 200:
                return resp.json()
            return {"error": resp.json().get("detail", resp.text)}
        except requests.RequestException as exc:
            return {"error": str(exc)}

    # ── Documents ───────────────────────────────────────────────────

    def list_documents(self) -> list[dict[str, Any]]:
        try:
            resp = self._request("GET", "/documents")
            if resp.status_code == 200:
                data = resp.json()
                return data.get("documents", [])
            return []
        except requests.RequestException:
            return []

    def delete_document(self, document_id: str) -> Optional[dict[str, Any]]:
        try:
            resp = self._request("DELETE", f"/documents/{document_id}")
            if resp.status_code == 200:
                return resp.json()
            return {"error": resp.json().get("detail", resp.text)}
        except requests.RequestException as exc:
            return {"error": str(exc)}


# ═══════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════

def init_session_state() -> None:
    """Initialize all session state variables."""
    if "api" not in st.session_state:
        st.session_state.api = None
    if "api_url" not in st.session_state:
        st.session_state.api_url = DEFAULT_API_URL
    if "connected" not in st.session_state:
        st.session_state.connected = False
    if "health_data" not in st.session_state:
        st.session_state.health_data = None
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_answer" not in st.session_state:
        st.session_state.current_answer = None
    if "upload_result" not in st.session_state:
        st.session_state.upload_result = None
    if "processing" not in st.session_state:
        st.session_state.processing = False


def connect_to_backend(api_url: str, api_key: Optional[str] = None) -> bool:
    """Attempt to connect to the OmniBrain backend."""
    api = OmniBrainAPI(api_url, api_key)
    health = api.health()
    if health and health.get("status") == "ok":
        st.session_state.api = api
        st.session_state.api_url = api_url
        st.session_state.connected = True
        st.session_state.health_data = health
        return True
    return False


def refresh_documents() -> None:
    """Refresh the document list from the backend."""
    if st.session_state.api:
        st.session_state.documents = st.session_state.api.list_documents()


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


# ═══════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════

def render_sidebar() -> None:
    """Render the sidebar with connection controls and document list."""
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-header">🧠 OmniBrain</div>',
            unsafe_allow_html=True,
        )

        # ── Connection settings ────────────────────────────────────
        with st.expander("🔗 Connection", expanded=not st.session_state.connected):
            api_url = st.text_input(
                "Backend URL",
                value=st.session_state.api_url,
                placeholder="http://localhost:8000",
                key="sidebar_api_url",
            )
            api_key = st.text_input(
                "API Key (optional)",
                value="",
                type="password",
                placeholder="Leave blank if disabled",
                key="sidebar_api_key",
            )

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔌 Connect", use_container_width=True, type="primary"):
                    with st.spinner("Connecting..."):
                        if connect_to_backend(
                            api_url, api_key if api_key else None
                        ):
                            st.success("✅ Connected!")
                            refresh_documents()
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("❌ Could not connect")
            with col2:
                if st.button("🔄 Refresh", use_container_width=True):
                    if st.session_state.api:
                        refresh_documents()
                        st.rerun()

        # ── Connection status ──────────────────────────────────────
        if st.session_state.connected:
            health = st.session_state.health_data
            if health:
                st.markdown(
                    f"""
                    <div style="background:#f0fdf4; border:1px solid #bbf7d0; 
                                border-radius:10px; padding:12px 16px; margin-bottom:16px;">
                        <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                            <span style="color:#16a34a; font-size:20px;">●</span>
                            <span style="font-weight:600; color:#166534;">Connected</span>
                        </div>
                        <div style="font-size:13px; color:#475569;">
                            📄 {health.get('documents_indexed', 0)} docs · 
                            🧩 {health.get('total_chunks', 0)} chunks · 
                            ⚡ {health.get('version', 'N/A')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ── Document list ──────────────────────────────────────
            st.markdown("### 📚 Documents")
            refresh_documents()
            docs = st.session_state.documents

            if not docs:
                st.info("No documents uploaded yet.")
            else:
                for doc in docs:
                    doc_id = doc.get("document_id", "")
                    filename = doc.get("filename", "Unknown")
                    pages = doc.get("pages", 0)
                    chunks = doc.get("chunks", 0)
                    size = format_file_size(doc.get("file_size_bytes", 0))

                    with st.container():
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(
                                f"""
                                <div style="background:#f8fafc; border-radius:8px; padding:10px 12px; 
                                            margin-bottom:6px; border:1px solid #e2e8f0;">
                                    <div style="font-weight:600; font-size:14px; color:#1e293b;
                                                overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                                        📄 {filename}
                                    </div>
                                    <div style="font-size:12px; color:#64748b; margin-top:4px;">
                                        {pages} pages · {chunks} chunks · {size}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        with col2:
                            if st.button("🗑️", key=f"del_{doc_id}", help=f"Delete {filename}"):
                                with st.spinner("Deleting..."):
                                    result = st.session_state.api.delete_document(doc_id)
                                    if result and "error" not in result:
                                        st.success(f"Deleted {filename}")
                                        refresh_documents()
                                        time.sleep(0.5)
                                        st.rerun()
                                    else:
                                        st.error(f"Delete failed: {result}")
        else:
            st.markdown(
                '<div style="background:#fef2f2; border:1px solid #fecaca; '
                'border-radius:10px; padding:12px 16px; text-align:center;">'
                '<span style="color:#dc2626; font-size:16px;">●</span> '
                '<span style="color:#991b1b; font-weight:500;">Not connected</span>'
                '<div style="font-size:12px; color:#64748b; margin-top:4px;">'
                'Enter the backend URL above and click Connect</div>'
                "</div>",
                unsafe_allow_html=True,
            )

        # ── Footer ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            '<div style="text-align:center; font-size:12px; color:#94a3b8;">'
            "OmniBrain v1.0 · Built with Streamlit</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════
# Main Content — Upload Tab
# ═══════════════════════════════════════════════════════════════════

def render_upload_tab() -> None:
    """Render the PDF upload section."""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>📤 Upload PDF</h3>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a PDF file to upload and index",
        type=["pdf"],
        label_visibility="collapsed",
        help="Upload a PDF document for AI-powered Q&A",
    )

    if uploaded_file is not None:
        st.markdown(
            f"""
            <div style="background:#f8fafc; border-radius:10px; padding:12px 16px; margin-top:8px;">
                <strong>📄 {uploaded_file.name}</strong> · {format_file_size(uploaded_file.size)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Save to a temp location and upload
        if st.button("🚀 Upload & Process", type="primary", use_container_width=True):
            if not st.session_state.connected or not st.session_state.api:
                st.error("❌ Please connect to the backend first.")
                return

            # Save uploaded file temporarily
            temp_dir = Path("/tmp/omnibrain_uploads")
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / uploaded_file.name

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("🔄 Processing document... This may take a moment."):
                st.session_state.processing = True
                result = st.session_state.api.upload(
                    str(temp_path), uploaded_file.name
                )

                # Clean up temp file
                try:
                    temp_path.unlink()
                except OSError:
                    pass

                st.session_state.processing = False

                if result and "error" not in result:
                    st.session_state.upload_result = result
                    st.balloons()
                    refresh_documents()
                    st.rerun()
                else:
                    st.error(
                        f"❌ Upload failed: {result.get('error', 'Unknown error')}"
                    )

    # ── Show upload result ───────────────────────────────────────
    if st.session_state.upload_result:
        result = st.session_state.upload_result
        st.markdown(
            f"""
            <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:12px; 
                        padding:16px 20px; margin-top:16px;">
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                    <span style="font-size:20px;">✅</span>
                    <span style="font-weight:600; color:#166534;">Upload Successful!</span>
                </div>
                <table style="width:100%; font-size:14px; color:#475569;">
                    <tr><td style="padding:4px 8px; font-weight:500;">Document ID</td>
                        <td style="padding:4px 8px;"><code>{result.get('document_id', 'N/A')[:16]}...</code></td></tr>
                    <tr><td style="padding:4px 8px; font-weight:500;">Filename</td>
                        <td style="padding:4px 8px;">{result.get('filename', 'N/A')}</td></tr>
                    <tr><td style="padding:4px 8px; font-weight:500;">Pages</td>
                        <td style="padding:4px 8px;">{result.get('pages', 0)}</td></tr>
                    <tr><td style="padding:4px 8px; font-weight:500;">Chunks</td>
                        <td style="padding:4px 8px;">{result.get('chunks', 0)}</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("✖️ Clear", key="clear_upload_result"):
            st.session_state.upload_result = None
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Stats summary ───────────────────────────────────────────
    if st.session_state.connected and st.session_state.health_data:
        health = st.session_state.health_data
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>📊 System Statistics</h3>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="value">{health.get('documents_indexed', 0)}</div>
                    <div class="label">Documents</div>
                </div>
                <div class="stat-box">
                    <div class="value">{health.get('total_chunks', 0)}</div>
                    <div class="label">Chunks</div>
                </div>
                <div class="stat-box">
                    <div class="value">{health.get('version', 'N/A')}</div>
                    <div class="label">Version</div>
                </div>
                <div class="stat-box">
                    <div class="value">{health.get('uptime_seconds', 0):.0f}s</div>
                    <div class="label">Uptime</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# Main Content — Chat Tab
# ═══════════════════════════════════════════════════════════════════

def render_chat_tab() -> None:
    """Render the Q&A chat interface."""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>💬 Ask Questions</h3>", unsafe_allow_html=True)

    # ── Document scope selector ─────────────────────────────────
    docs = st.session_state.documents
    selected_docs: Optional[list[str]] = None

    if docs:
        st.markdown(
            '<div style="font-size:13px; color:#64748b; margin-bottom:8px;">'
            "Search scope (optional — leave empty to search all documents):"
            "</div>",
            unsafe_allow_html=True,
        )
        doc_options = {f"{d['filename']} ({d['document_id'][:8]}...)": d["document_id"] for d in docs}
        selected_names = st.multiselect(
            "Filter by document",
            options=list(doc_options.keys()),
            placeholder="All documents",
            label_visibility="collapsed",
        )
        if selected_names:
            selected_docs = [doc_options[name] for name in selected_names]

        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            col1, col2 = st.columns(2)
            with col1:
                top_k = st.slider(
                    "Top-K chunks", min_value=1, max_value=20, value=5,
                    help="Number of relevant chunks to retrieve",
                )
            with col2:
                temperature = st.slider(
                    "Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05,
                    help="Higher = more creative, lower = more factual",
                )
    else:
        st.info("📭 No documents available. Upload a PDF first to ask questions.")
        top_k = 5
        temperature = 0.2

    # ── Question input ──────────────────────────────────────────
    question = st.text_input(
        "Your question",
        placeholder="e.g., What is the main finding of the report?",
        label_visibility="collapsed",
        key="question_input",
        disabled=not st.session_state.connected or not docs,
    )

    col1, col2 = st.columns([5, 1])
    with col1:
        ask_button = st.button(
            "🎯 Ask OmniBrain",
            type="primary",
            use_container_width=True,
            disabled=not question or not st.session_state.connected or not docs,
        )
    with col2:
        clear_btn = st.button(
            "✖️ Clear",
            use_container_width=True,
            disabled=not st.session_state.current_answer,
        )

    # ── Process the question ────────────────────────────────────
    if ask_button and question:
        if not st.session_state.api:
            st.error("❌ Not connected to backend.")
        else:
            with st.spinner("🧠 Thinking... Retrieving context & generating answer..."):
                result = st.session_state.api.chat(
                    question=question,
                    document_ids=selected_docs,
                    top_k=top_k,
                    temperature=temperature if docs else None,
                )

                if result and "error" not in result:
                    st.session_state.current_answer = {
                        "question": question,
                        "answer": result.get("answer", ""),
                        "sources": result.get("sources", []),
                        "processing_time_ms": result.get("processing_time_ms", 0),
                        "model_used": result.get("model_used", ""),
                    }
                    # Add to history
                    st.session_state.chat_history.append(
                        {
                            "role": "user",
                            "content": question,
                            "timestamp": time.strftime("%H:%M:%S"),
                        }
                    )
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": result.get("answer", ""),
                            "sources": result.get("sources", []),
                            "processing_time_ms": result.get("processing_time_ms", 0),
                            "model_used": result.get("model_used", ""),
                            "timestamp": time.strftime("%H:%M:%S"),
                        }
                    )
                    st.rerun()
                else:
                    st.error(
                        f"❌ Error: {result.get('error', 'Unknown error')}"
                    )

    if clear_btn:
        st.session_state.current_answer = None
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Display current answer ──────────────────────────────────
    if st.session_state.current_answer:
        answer_data = st.session_state.current_answer

        # The answer box
        st.markdown(
            f"""
            <div class="answer-box">
                <h4>🤖 Answer</h4>
                <div style="font-size:15px; color:#1e293b; white-space:pre-wrap;">
                    {answer_data['answer']}
                </div>
                <div style="margin-top:12px; font-size:12px; color:#94a3b8; 
                            border-top:1px solid #e2e8f0; padding-top:10px;">
                    Model: {answer_data.get('model_used', 'N/A')} · 
                    Processed in {answer_data.get('processing_time_ms', 0)}ms
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Sources
        sources = answer_data.get("sources", [])
        if sources:
            st.markdown(
                f"<h4 style='margin-bottom:10px;'>📎 Sources ({len(sources)})</h4>",
                unsafe_allow_html=True,
            )
            for i, src in enumerate(sources):
                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="filename">
                            📄 {src.get('filename', 'Unknown')}
                        </div>
                        <div class="meta">
                            Page {src.get('page_number', 'N/A')} · 
                            Chunk #{src.get('chunk_index', 'N/A')} · 
                            <span class="score">{src.get('similarity_score', 0):.3f} similarity</span>
                        </div>
                        <div class="snippet">"{src.get('snippet', '')[:300]}..."</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ── Chat history (collapsed) ────────────────────────────────
    if st.session_state.chat_history:
        with st.expander("📜 Chat History", expanded=False):
            for entry in st.session_state.chat_history:
                if entry["role"] == "user":
                    st.markdown(
                        f"""
                        <div style="background:#eef2ff; border-radius:10px; padding:10px 14px; 
                                    margin-bottom:8px; border:1px solid #dde4ff;">
                            <div style="font-weight:600; color:#4338ca; font-size:13px; margin-bottom:4px;">
                                🙋 You · {entry.get('timestamp', '')}
                            </div>
                            <div style="color:#1e293b;">{entry['content']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="background:#f8fafc; border-radius:10px; padding:10px 14px; 
                                    margin-bottom:8px; border:1px solid #e2e8f0;">
                            <div style="font-weight:600; color:#6366f1; font-size:13px; margin-bottom:4px;">
                                🧠 OmniBrain · {entry.get('timestamp', '')} · 
                                {entry.get('processing_time_ms', 0)}ms
                            </div>
                            <div style="color:#1e293b;">{entry['content'][:500]}{'...' if len(entry['content']) > 500 else ''}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.current_answer = None
                st.rerun()


# ═══════════════════════════════════════════════════════════════════
# Main App
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point for the Streamlit app."""
    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Initialize session state
    init_session_state()

    # Render sidebar
    render_sidebar()

    # ── Main header ─────────────────────────────────────────────
    st.markdown(
        """
        <div class="main-header">
            <span style="font-size:32px;">🧠</span>
            <h1>OmniBrain</h1>
            <span class="badge">AI Document Q&A</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p style="color:#64748b; font-size:15px; margin-bottom:24px;">'
        "Upload PDF documents and ask natural-language questions. "
        "OmniBrain uses AI to find relevant content and provide accurate answers with citations.</p>",
        unsafe_allow_html=True,
    )

    # ── Tabs ────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["📤 Upload & Stats", "💬 Q&A Chat"])

    with tab1:
        render_upload_tab()

    with tab2:
        if not st.session_state.connected:
            st.warning("🔌 Please connect to the backend using the sidebar first.")
        else:
            render_chat_tab()


if __name__ == "__main__":
    main()