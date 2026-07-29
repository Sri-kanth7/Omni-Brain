import streamlit as st
import requests
import os
import time

# Config page
st.set_page_config(
    page_title="OmniBrain Orchestrator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, premium dark-mode styling
st.markdown("""
<style>
    /* Dark Theme Base Colors */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Header layout */
    .main-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #1d4ed8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Premium components cards */
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #60a5fa;
    }
    
    /* Badge colors */
    .badge-success {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-error {
        background-color: #7f1d1d;
        color: #fca5a5;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    /* Prompt suggestion bubbles */
    .prompt-bubble {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px 15px;
        margin: 5px 0;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
    }
    .prompt-bubble:hover {
        background-color: #334155;
        border-color: #3b82f6;
    }
    
    /* Agent step timeline cards */
    .step-card {
        background-color: #111827;
        border-left: 4px solid #3b82f6;
        padding: 12px 18px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
    }
    .citation-tag {
        background-color: #1e3a8a;
        color: #93c5fd;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-right: 5px;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)

# API Endpoint URL
API_URL = "http://127.0.0.1:8000"

# Main UI title
st.markdown("<div class='main-title'>🧠 OMNIBRAIN</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Agentic Multi-Modal RAG Orchestrator for Financial Workflows</div>", unsafe_allow_html=True)

# Fetch API status
try:
    status_resp = requests.get(f"{API_URL}/status").json()
    api_online = True
except Exception:
    api_online = False
    status_resp = {"text_chunks": 0, "image_chunks": 0, "environment": "offline"}

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/brain.png", width=64)
    st.markdown("### Control Panel")
    
    # Connection details
    if api_online:
        st.success("API Server: Connected")
    else:
        st.error("API Server: Offline (Port 8000)")
        
    st.info(f"Vector Database Mode: **{status_resp.get('environment', 'N/A').upper()}**")
    
    # Document uploader
    st.markdown("---")
    st.markdown("#### Upload Financial Document (PDF)")
    uploaded_file = st.file_uploader("Choose a PDF report...", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Ingest Report"):
            with st.spinner("Uploading and triggers background parsing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    resp = requests.post(f"{API_URL}/upload", files=files).json()
                    st.success(f"Successfully uploaded: {resp.get('filename')}")
                    st.info("Ingesting text chunks and extracting charts in the background. Refresh status below.")
                    time.sleep(1.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"Upload failed: {e}")
                    
    # Vector store status info
    st.markdown("---")
    st.markdown("#### Vector Database Size")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Text Blocks", status_resp.get("text_chunks", 0))
    with col2:
        st.metric("Charts", status_resp.get("image_chunks", 0))

# Preset questions
st.markdown("### Ask a question about the report:")
preset_queries = [
    "What is the average stock price of AAPL? Check general earnings report and the balance_sheet.png chart too.",
    "Give me maximum high price of MSFT and summarize latest market notes.",
    "Summarize Microsoft cloud division growth and list references.",
]

# Create columns for quick prompt selections
cols = st.columns(3)
selected_query = ""
for idx, q in enumerate(preset_queries):
    if cols[idx].button(f"📄 Q{idx+1}: {q[:45]}...", help=q):
        selected_query = q

# Query Input
query_val = selected_query if selected_query else ""
user_query = st.text_input("Enter your query:", value=query_val, placeholder="e.g. What is the average stock price of AAPL? Check balance sheet image too.")

if st.button("Submit Query", type="primary") or selected_query:
    if not user_query.strip():
        st.warning("Please enter a question.")
    elif not api_online:
        st.error("FastAPI server must be running to process queries. Run uvicorn server on port 8000.")
    else:
        with st.spinner("Processing through input guardrails and LangGraph state machine..."):
            try:
                # Query API
                payload = {"query": user_query}
                resp = requests.post(f"{API_URL}/query", json=payload).json()
                
                # Check guardrails status
                if not resp.get("allowed", True):
                    st.markdown(f"### 🛡️ Guardrail Refusal")
                    st.error(resp.get("answer"))
                else:
                    st.markdown("### 📊 Generated Memo")
                    st.markdown(resp.get("answer"))
                    
                    # Display associated images/charts if retrieved
                    img_path = resp.get("image_path")
                    if img_path:
                        st.markdown("#### 🖼️ Retrieved Context Visualization")
                        # API serves images at API_URL/static/extracted_images/name
                        # Our ingestion returns image_path relative to project root or static folder
                        img_filename = os.path.basename(img_path)
                        img_url = f"{API_URL}/static/extracted_images/{img_filename}"
                        st.image(img_url, caption=f"Extracted balance sheet analysis: {img_filename}", use_container_width=True)
                    
                    # Collapsible section for Agent steps/thought process
                    st.markdown("---")
                    with st.expander("⛓️ LangGraph Orchestrator Execution steps", expanded=True):
                        steps = resp.get("intermediate_steps", [])
                        if not steps:
                            st.write("Executed directly.")
                        else:
                            for idx, s in enumerate(steps):
                                st.markdown(f"<div class='step-card'><b>Step {idx+1}</b>: Agent routed to <b><code>{s}</code></b></div>", unsafe_allow_html=True)
                                
                    # Collapsible section for citations
                    citations = resp.get("citations", [])
                    if citations:
                        with st.expander("📚 Citations & Documents"):
                            for c in citations:
                                source = c.get("source", "Unknown")
                                page_info = f" (Page {c.get('page')})" if c.get("page") else ""
                                st.markdown(f"<span class='citation-tag'>📄 {source}{page_info}</span>", unsafe_allow_html=True)
                                
                    # Collapsible section for Evaluation metrics
                    eval_data = resp.get("evaluation", {})
                    if eval_data:
                        with st.expander("📈 Observer & Evaluation Telemetry (Langfuse)"):
                            status_label = eval_data.get("status", "PASSED").upper()
                            status_badge = f"<span class='badge-success'>{status_label}</span>" if status_label == "PASSED" else f"<span class='badge-error'>{status_label}</span>"
                            
                            st.markdown(f"**Verification Status**: {status_badge}", unsafe_allow_html=True)
                            
                            c1, c2, c3 = st.columns(3)
                            c1.markdown(f"<div class='metric-card'>Groundedness<br><span class='metric-value'>{eval_data.get('groundedness', 0.0):.2f}</span></div>", unsafe_allow_html=True)
                            c2.markdown(f"<div class='metric-card'>Relevance<br><span class='metric-value'>{eval_data.get('relevance', 0.0):.2f}</span></div>", unsafe_allow_html=True)
                            c3.markdown(f"<div class='metric-card'>Hallucination Score<br><span class='metric-value'>{eval_data.get('hallucination_score', 0.0):.2f}</span></div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Failed to query backend: {e}")
