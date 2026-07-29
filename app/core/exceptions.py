"""
Custom exception hierarchy for OmniBrain.
Each exception maps to a specific HTTP status code.
"""

from typing import Any, Optional


class OmniBrainError(Exception):
    """Base exception for all OmniBrain errors."""

    def __init__(
        self,
        message: str = "An unexpected error occurred",
        status_code: int = 500,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class DocumentNotFoundError(OmniBrainError):
    """Raised when a requested document does not exist."""

    def __init__(self, document_id: str) -> None:
        super().__init__(
            message=f"Document '{document_id}' not found",
            status_code=404,
            details={"document_id": document_id},
        )


class InvalidFileTypeError(OmniBrainError):
    """Raised when an uploaded file has an unsupported extension."""

    def __init__(self, filename: str, allowed: set[str]) -> None:
        super().__init__(
            message=f"File '{filename}' has an unsupported type. Allowed: {', '.join(allowed)}",
            status_code=400,
            details={"filename": filename, "allowed_extensions": list(allowed)},
        )


class FileTooLargeError(OmniBrainError):
    """Raised when an upload exceeds the size limit."""

    def __init__(self, size_mb: float, max_mb: int) -> None:
        super().__init__(
            message=f"File size {size_mb:.1f} MB exceeds the {max_mb} MB limit",
            status_code=413,
            details={"file_size_mb": size_mb, "max_size_mb": max_mb},
        )


class ProcessingError(OmniBrainError):
    """Raised when document processing fails."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message=message, status_code=500, details=details)


class GeminiAPIError(OmniBrainError):
    """Raised when the Gemini API call fails."""

    def __init__(self, message: str = "Gemini API request failed") -> None:
        super().__init__(message=message, status_code=502)


class AuthenticationError(OmniBrainError):
    """Raised when authentication fails."""

    def __init__(self) -> None:
        super().__init__(message="Invalid or missing API key", status_code=401)