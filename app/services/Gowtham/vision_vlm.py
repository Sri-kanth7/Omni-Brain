import base64
import io
import json
from typing import List, Dict, Any, Optional, Literal
from PIL import Image
import fitz  # PyMuPDF
from pydantic import BaseModel, Field

# Import configurations
import config

# Define the structured output format for charts and tables
class VisualExtraction(BaseModel):
    type: Literal["chart", "table", "diagram", "other"] = Field(
        description="The category of visual elements found: chart, table, diagram, or other"
    )
    description: str = Field(
        description="A concise but detailed summary of what the visual data shows, including main trends, titles, or insights."
    )
    extracted_values: List[Dict[str, Any]] = Field(
        description="A list of key-value pairs of the data extracted. For tables: list of rows as dictionaries. For charts: list of data points with x/y values."
    )
    axis_labels: Optional[List[str]] = Field(
        default=None,
        description="For charts, list the labels or names of the axes (e.g. ['Year', 'Revenue in USD']). For tables, list the column names."
    )

def image_to_base64(image_bytes: bytes) -> str:
    """Converts image bytes to base64 encoded string."""
    return base64.b64encode(image_bytes).decode("utf-8")

def render_pdf_page_to_bytes(pdf_path: str, page_number: int, dpi: int = 150) -> bytes:
    """Renders a PDF page to PNG bytes."""
    doc = fitz.open(pdf_path)
    if page_number < 0 or page_number >= len(doc):
        raise ValueError(f"Page number {page_number} is out of bounds for PDF with {len(doc)} pages.")
    
    page = doc.load_page(page_number)
    # Set resolution matrix
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes

def extract_images_from_pdf_page(pdf_path: str, page_number: int) -> List[bytes]:
    """Extracts raw embedded images from a specific page in the PDF."""
    doc = fitz.open(pdf_path)
    images = []
    
    if page_number < 0 or page_number >= len(doc):
        doc.close()
        return []
        
    page = doc.load_page(page_number)
    image_list = page.get_images(full=True)
    
    for img_index, img in enumerate(image_list):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        images.append(image_bytes)
        
    doc.close()
    return images

def analyze_visual_element(image_bytes: bytes, provider: Optional[str] = None) -> VisualExtraction:
    """Sends an image to the VLM and extracts structured information according to VisualExtraction schema."""
    selected_provider = provider or config.DEFAULT_PROVIDER
    api_key = config.get_api_key(selected_provider)
    vlm_model = config.get_vlm_model(selected_provider)
    
    base64_img = image_to_base64(image_bytes)
    
    # Prompt instructing the VLM to extract structured data
    system_prompt = (
        "You are an expert financial and data analyst AI. Analyze the provided image of a chart or table "
        "and return structured details. Extract exact data values, describe trends, and identify labels."
    )
    
    # Multimodal human message payload
    from langchain_core.messages import HumanMessage
    message_content = [
        {"type": "text", "text": system_prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_img}"}
        }
    ]
    message = HumanMessage(content=message_content)
    
    if selected_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Check API key presence
        if not api_key:
            # If missing key, fall back to mock extraction so it remains runnable without credentials
            print("WARNING: GEMINI_API_KEY environment variable is missing. Running in mock/dry-run mode.")
            return get_mock_extraction()
            
        llm = ChatGoogleGenerativeAI(
            model=vlm_model,
            google_api_key=api_key,
            temperature=0.0
        )
    elif selected_provider == "openai":
        from langchain_openai import ChatOpenAI
        if not api_key:
            print("WARNING: OPENAI_API_KEY environment variable is missing. Running in mock/dry-run mode.")
            return get_mock_extraction()
            
        llm = ChatOpenAI(
            model=vlm_model,
            api_key=api_key,
            temperature=0.0
        )
    else:
        raise ValueError(f"Unsupported provider: {selected_provider}")
        
    # Bind structured output schema using LangChain with_structured_output
    structured_llm = llm.with_structured_output(VisualExtraction)
    
    try:
        result = structured_llm.invoke([message])
        return result
    except Exception as e:
        print(f"Error during VLM inference: {e}. Falling back to standard JSON parsing...")
        # Fallback raw parse if structured fails
        try:
            raw_llm = llm.invoke([message])
            # Attempt to parse json from raw output
            content = raw_llm.content
            # Strip markdown json block wrappers if any
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            data = json.loads(content)
            return VisualExtraction(**data)
        except Exception as fallback_err:
            print(f"Fallback extraction failed: {fallback_err}. Returning mock extraction.")
            return get_mock_extraction()

def get_mock_extraction() -> VisualExtraction:
    """Returns a realistic mock extraction when API keys are not available."""
    return VisualExtraction(
        type="chart",
        description="[MOCK DATA] A line chart showing corporate revenue trends between 2022 and 2024. Revenue rises steadily.",
        extracted_values=[
            {"Year": 2022, "Revenue_Millions": 2.0},
            {"Year": 2023, "Revenue_Millions": 3.5},
            {"Year": 2024, "Revenue_Millions": 5.0}
        ],
        axis_labels=["Year", "Revenue_Millions"]
    )
