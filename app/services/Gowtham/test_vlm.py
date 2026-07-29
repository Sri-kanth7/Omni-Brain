import os
import fitz  # PyMuPDF
import vision_vlm
import config

def create_sample_pdf(pdf_path: str):
    """Creates a sample PDF with visual elements (tables/shapes) for testing."""
    print(f"Generating programmatic test PDF at: {pdf_path}...")
    doc = fitz.open()
    
    # Page 1: Revenue Line Chart Mock
    page1 = doc.new_page()
    page1.insert_text(fitz.Point(50, 50), "OmniBrain Analysis Report - Page 1", fontsize=18, fontname="helv", color=(0.1, 0.1, 0.1))
    page1.insert_text(fitz.Point(50, 80), "Figure 1: Corporate Revenue Growth (2022-2024)", fontsize=14, fontname="helv", color=(0.2, 0.2, 0.2))
    
    # Draw simple line chart axes and data line
    page1.draw_line(fitz.Point(100, 250), fitz.Point(400, 250), color=(0.5, 0.5, 0.5), width=2) # X Axis
    page1.draw_line(fitz.Point(100, 100), fitz.Point(100, 250), color=(0.5, 0.5, 0.5), width=2) # Y Axis
    # Data line: 2022 ($2M) -> 2023 ($3.5M) -> 2024 ($5M)
    # Mapping: 2022 (x=120, y=230), 2023 (x=240, y=170), 2024 (x=360, y=110)
    page1.draw_line(fitz.Point(120, 230), fitz.Point(240, 170), color=(0, 0, 1), width=3) # Blue line
    page1.draw_line(fitz.Point(240, 170), fitz.Point(360, 110), color=(0, 0, 1), width=3)
    
    # Draw axis ticks
    page1.insert_text(fitz.Point(120, 270), "2022", fontsize=10)
    page1.insert_text(fitz.Point(240, 270), "2023", fontsize=10)
    page1.insert_text(fitz.Point(360, 270), "2024", fontsize=10)
    page1.insert_text(fitz.Point(60, 230), "$2M", fontsize=10)
    page1.insert_text(fitz.Point(60, 170), "$3.5M", fontsize=10)
    page1.insert_text(fitz.Point(60, 110), "$5M", fontsize=10)
    
    # Page 2: Financial Grid Table
    page2 = doc.new_page()
    page2.insert_text(fitz.Point(50, 50), "OmniBrain Analysis Report - Page 2", fontsize=18, fontname="helv", color=(0.1, 0.1, 0.1))
    page2.insert_text(fitz.Point(50, 80), "Table 1: Key Metrics Comparison", fontsize=14, fontname="helv", color=(0.2, 0.2, 0.2))
    
    # Draw table border and lines
    page2.draw_rect(fitz.Rect(50, 110, 450, 230), color=(0.3, 0.3, 0.3), width=1)
    page2.draw_line(fitz.Point(50, 140), fitz.Point(450, 140), color=(0.3, 0.3, 0.3), width=1) # Header separator
    page2.draw_line(fitz.Point(50, 170), fitz.Point(450, 170), color=(0.3, 0.3, 0.3), width=1)
    page2.draw_line(fitz.Point(50, 200), fitz.Point(450, 200), color=(0.3, 0.3, 0.3), width=1)
    
    # Column lines
    page2.draw_line(fitz.Point(180, 110), fitz.Point(180, 230), color=(0.3, 0.3, 0.3), width=1)
    page2.draw_line(fitz.Point(310, 110), fitz.Point(310, 230), color=(0.3, 0.3, 0.3), width=1)
    
    # Table Header text
    page2.insert_text(fitz.Point(60, 130), "Metric", fontsize=11, fontname="hebo")
    page2.insert_text(fitz.Point(190, 130), "Year 2023", fontsize=11, fontname="hebo")
    page2.insert_text(fitz.Point(320, 130), "Year 2024", fontsize=11, fontname="hebo")
    
    # Table Row 1
    page2.insert_text(fitz.Point(60, 160), "Revenue", fontsize=10)
    page2.insert_text(fitz.Point(190, 160), "$3.5M", fontsize=10)
    page2.insert_text(fitz.Point(320, 160), "$5.0M", fontsize=10)
    
    # Table Row 2
    page2.insert_text(fitz.Point(60, 190), "Profit Margin", fontsize=10)
    page2.insert_text(fitz.Point(190, 190), "12%", fontsize=10)
    page2.insert_text(fitz.Point(320, 190), "15%", fontsize=10)
    
    # Table Row 3
    page2.insert_text(fitz.Point(60, 220), "Employee Count", fontsize=10)
    page2.insert_text(fitz.Point(190, 220), "45", fontsize=10)
    page2.insert_text(fitz.Point(320, 220), "62", fontsize=10)
    
    doc.save(pdf_path)
    doc.close()
    print("Test PDF successfully created.\n")

def run_tests():
    pdf_filename = "test_document.pdf"
    
    # Create sample PDF if it doesn't exist
    if not os.path.exists(pdf_filename):
        create_sample_pdf(pdf_filename)
        
    config.print_status()
    
    # Render and analyze Page 1 (Line Chart)
    print("\n[VLM Test] Rendering and analyzing Page 1 (Line Chart)...")
    try:
        page_1_bytes = vision_vlm.render_pdf_page_to_bytes(pdf_filename, page_number=0)
        extraction_1 = vision_vlm.analyze_visual_element(page_1_bytes)
        
        print("\n--- Page 1 VLM Extraction JSON Output ---")
        print(extraction_1.model_dump_json(indent=2))
    except Exception as e:
        print(f"Failed page 1 VLM extraction test: {e}")
        
    # Render and analyze Page 2 (Financial Table)
    print("\n[VLM Test] Rendering and analyzing Page 2 (Financial Table)...")
    try:
        page_2_bytes = vision_vlm.render_pdf_page_to_bytes(pdf_filename, page_number=1)
        extraction_2 = vision_vlm.analyze_visual_element(page_2_bytes)
        
        print("\n--- Page 2 VLM Extraction JSON Output ---")
        print(extraction_2.model_dump_json(indent=2))
    except Exception as e:
        print(f"Failed page 2 VLM extraction test: {e}")

if __name__ == "__main__":
    run_tests()
