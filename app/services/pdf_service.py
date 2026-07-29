"""
PDF rendering, preprocessing, text extraction, OCR, and chunking.

Sowbarnika's Module:
- PDF rendering & preprocessing
- Text extraction (with optional OCR fallback via pytesseract)
- Configurable chunking pipeline
"""

import io
import re
from pathlib import Path
from typing import Any, Optional

import PyPDF2
import pytesseract
from pdf2image import convert_from_path

from app.core.config import settings
from app.core.exceptions import ProcessingError
from app.core.logger import get_logger

logger = get_logger(__name__)


class PDFService:
    """
    Handles all PDF-related operations including text extraction,
    preprocessing, optional OCR support, and document chunking.
    """

    # ── Text Extraction ──────────────────────────────────────────────

    def extract_text(self, file_path: str) -> dict[str, Any]:
        """
        Extract text from a PDF file.

        First attempts PyPDF2 extraction. If the extracted text is too sparse,
        falls back to OCR via pytesseract.

        Args:
            file_path: Absolute path to the PDF file.

        Returns:
            Dict with keys:
                - pages (list[str]): text content per page
                - total_pages (int)
                - method_used (str): "pypdf2" or "ocr"
                - metadata (dict): PDF metadata (title, author, etc.)

        Raises:
            ProcessingError: If the file cannot be read or decoded.
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise ProcessingError(f"File not found: {file_path}")

        logger.info(f"Extracting text from: {file_path}")

        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                metadata = {
                    "title": reader.metadata.get("/Title", ""),
                    "author": reader.metadata.get("/Author", ""),
                    "subject": reader.metadata.get("/Subject", ""),
                    "creator": reader.metadata.get("/Creator", ""),
                }
                pages: list[str] = []
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    pages.append(text)

                total_text = " ".join(pages)
                method_used = "pypdf2"

                # Fall back to OCR if extracted text is too sparse
                if len(total_text.strip()) < 50 and len(pages) > 0:
                    logger.info("PyPDF2 text too sparse; falling back to OCR")
                    pages = self._ocr_extract(file_path)
                    method_used = "ocr"

                logger.info(
                    f"Extraction complete | method={method_used} "
                    f"| pages={len(pages)} | total_chars={sum(len(p) for p in pages)}"
                )

                return {
                    "pages": pages,
                    "total_pages": len(pages),
                    "method_used": method_used,
                    "metadata": metadata,
                }

        except PyPDF2.errors.PdfReadError as exc:
            raise ProcessingError(f"Failed to read PDF: {exc}") from exc
        except Exception as exc:
            raise ProcessingError(f"Unexpected error during PDF extraction: {exc}") from exc

    def _ocr_extract(self, file_path: str) -> list[str]:
        """
        Extract text from PDF using OCR (tesseract).

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of page text strings.
        """
        logger.info(f"Running OCR on: {file_path}")
        try:
            images = convert_from_path(file_path, dpi=300)
            pages: list[str] = []
            for i, img in enumerate(images):
                text = pytesseract.image_to_string(img, lang="eng")
                pages.append(text)
                logger.debug(f"OCR page {i + 1}/{len(images)}: {len(text)} chars")
            return pages
        except Exception as exc:
            raise ProcessingError(f"OCR extraction failed: {exc}") from exc

    # ── Text Preprocessing & Cleaning ────────────────────────────────

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.

        Steps:
            - Remove null bytes
            - Replace multiple newlines with a single newline
            - Replace multiple spaces with a single space
            - Remove orphaned characters
            - Strip leading/trailing whitespace per line

        Args:
            text: Raw extracted text.

        Returns:
            Cleaned text string.
        """
        if not text:
            return ""

        # Remove null bytes
        text = text.replace("\x00", "")

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse excessive newlines (keep paragraph breaks)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Collapse multiple spaces
        text = re.sub(r" {2,}", " ", text)

        # Remove orphaned single characters on their own line (likely artifacts)
        text = re.sub(r"\n\s*[A-Za-z0-9]\s*\n", "\n", text)

        # Strip whitespace per line and remove empty lines at start/end
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text.strip()

    def preprocess_pages(
        self, pages: list[str], metadata: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """
        Preprocess a list of page texts: clean and structure with metadata.

        Args:
            pages: Raw page texts from extract_text().
            metadata: Optional document-level metadata to attach.

        Returns:
            List of dicts with keys: page_number, raw_text, cleaned_text, char_count.
        """
        processed: list[dict[str, Any]] = []
        for i, raw_text in enumerate(pages):
            cleaned = self.clean_text(raw_text)
            processed.append({
                "page_number": i + 1,
                "raw_text": raw_text,
                "cleaned_text": cleaned,
                "char_count": len(cleaned),
            })
        logger.debug(f"Preprocessed {len(processed)} pages")
        return processed

    # ── Chunking Pipeline ────────────────────────────────────────────

    def chunk_document(
        self,
        pages: list[dict[str, Any]],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Split a preprocessed document into overlapping chunks.

        Uses a sliding-window approach over the concatenated cleaned text.
        Each chunk retains its source page numbers for citation.

        Args:
            pages: Preprocessed pages from preprocess_pages().
            chunk_size: Max characters per chunk (default: settings.CHUNK_SIZE).
            chunk_overlap: Overlap between chunks (default: settings.CHUNK_OVERLAP).

        Returns:
            List of chunk dicts with keys:
                - chunk_index (int)
                - text (str)
                - page_numbers (list[int])
                - char_count (int)
        """
        chunk_size = chunk_size or settings.CHUNK_SIZE
        chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        # Build a page-indexed text stream
        page_map: list[dict[str, Any]] = []
        for page in pages:
            text = page["cleaned_text"]
            if text:
                page_map.append({
                    "page_number": page["page_number"],
                    "text": text,
                })

        if not page_map:
            logger.warning("No text found to chunk")
            return []

        # Concatenate all text with page markers
        full_text = ""
        page_boundaries: list[dict[str, Any]] = []
        char_offset = 0
        for entry in page_map:
            start = char_offset
            full_text += entry["text"] + "\n"
            char_offset = len(full_text)
            page_boundaries.append({
                "page_number": entry["page_number"],
                "start": start,
                "end": char_offset,
            })

        # Sliding-window chunking
        chunks: list[dict[str, Any]] = []
        start = 0
        chunk_index = 0

        while start < len(full_text):
            end = min(start + chunk_size, len(full_text))

            # Try to break at a sentence boundary for clean chunks
            if end < len(full_text):
                # Look for period + space or newline within the last 20% of the chunk
                search_start = max(start, end - int(chunk_size * 0.2))
                break_point = max(
                    full_text.rfind(". ", search_start, end),
                    full_text.rfind("\n", search_start, end),
                )
                if break_point > search_start:
                    end = break_point + 1

            chunk_text = full_text[start:end].strip()
            if chunk_text:
                # Determine source page numbers for this chunk
                chunk_pages = []
                for boundary in page_boundaries:
                    if boundary["start"] < end and boundary["end"] > start:
                        chunk_pages.append(boundary["page_number"])

                chunks.append({
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "page_numbers": sorted(set(chunk_pages)),
                    "char_count": len(chunk_text),
                })
                chunk_index += 1

            # Slide the window
            start = end - chunk_overlap if end < len(full_text) else len(full_text)

        logger.info(
            f"Chunking complete | {len(chunks)} chunks "
            f"| size={chunk_size} | overlap={chunk_overlap}"
        )
        return chunks