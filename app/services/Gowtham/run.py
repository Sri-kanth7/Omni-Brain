import os
import re
import fitz  # PyMuPDF
import config
from retriever import SimpleInMemoryVectorStore, Document
from self_rag import SelfRAGPipeline
import vision_vlm

def seed_database(store: SimpleInMemoryVectorStore, pdf_path: str):
    """Parses text pages from the PDF and indexes them into the vector store."""
    print(f"Ingesting text pages from PDF: {pdf_path}...")
    doc = fitz.open(pdf_path)
    docs = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": pdf_path, "page": page_num + 1}
            ))
    store.add_documents(docs)
    doc.close()
    print(f"Successfully indexed {len(docs)} text pages.\n")

def run_interactive_session():
    # Print configuration status
    config.print_status()
    
    # Get PDF path from user
    pdf_path = input("Enter the path to your PDF file (or press Enter to use test_document.pdf): ").strip()
    if not pdf_path:
        pdf_path = "test_document.pdf"
        if not os.path.exists(pdf_path):
            from test_vlm import create_sample_pdf
            create_sample_pdf(pdf_path)
            
    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' does not exist.")
        return
        
    # 1. Initialize retriever and index PDF
    store = SimpleInMemoryVectorStore()
    seed_database(store, pdf_path)
    
    # 2. Build Self-RAG Pipeline
    pipeline = SelfRAGPipeline(store)
    
    print("\n=======================================================")
    print("OmniBrain Interactive Q&A Session Started!")
    print("Ask any question about your PDF (textual or visual charts/tables).")
    print("Type 'exit' to end the session.")
    print("=======================================================")
    
    while True:
        query = input("\nAsk a question: ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            print("Ending Q&A Session. Goodbye!")
            break
            
        # Router: Check if question is visual or refers to a chart/table
        visual_keywords = ["chart", "table", "graph", "figure", "visual", "image", "diagram", "page"]
        is_visual = any(kw in query.lower() for kw in visual_keywords)
        
        if is_visual:
            # Attempt to extract page number from query (e.g. "page 1" -> page_number=0)
            page_match = re.search(r'page\s*(\d+)', query.lower())
            target_page = None
            if page_match:
                target_page = int(page_match.group(1)) - 1
                print(f"--> Router: Query refers to page {target_page + 1}. Routing to Vision Agent.")
            else:
                # Scrape database for pages containing "table", "figure" or "chart"
                print("--> Router: Visual keyword detected. Searching for table/chart pages...")
                for doc_obj in store.documents:
                    content_lower = doc_obj.page_content.lower()
                    if "table" in content_lower or "figure" in content_lower or "chart" in content_lower:
                        target_page = doc_obj.metadata.get("page", 1) - 1
                        break
                if target_page is None:
                    target_page = 0  # Default to page 1
                print(f"--> Router: Selected page {target_page + 1} based on keyword match. Routing to Vision Agent.")
                
            # Execute Vision VLM
            try:
                print(f"--- [Vision Agent Action] Rendering page {target_page + 1}... ---")
                page_bytes = vision_vlm.render_pdf_page_to_bytes(pdf_path, page_number=target_page)
                
                print(f"--- [Vision Agent Action] Sending page image to VLM... ---")
                extraction = vision_vlm.analyze_visual_element(page_bytes)
                
                print(f"\n[Vision Agent Response]:")
                print(f"Detected Type: {extraction.type.upper()}")
                print(f"Description: {extraction.description}")
                if extraction.extracted_values:
                    print("Extracted Structured Data:")
                    for val in extraction.extracted_values:
                        print(f"  - {val}")
                if extraction.axis_labels:
                    print(f"Labels: {extraction.axis_labels}")
            except Exception as e:
                print(f"Vision Agent Error: {e}")
        else:
            # General textual question -> Route to Self-RAG
            print("--> Router: Textual query. Routing to Self-RAG pipeline.")
            try:
                result = pipeline.run(query)
                print(f"\n[Self-RAG Response]:")
                print(result["generation"])
            except Exception as e:
                print(f"Self-RAG Error: {e}")

if __name__ == "__main__":
    run_interactive_session()
