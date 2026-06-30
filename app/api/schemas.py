from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class JobStatus(str, Enum):
    QUEUED = "queued"
    OCR_PROCESSING = "ocr_processing"
    FIELD_EXTRACTION = "field_extraction"
    TABLE_EXTRACTION = "table_extraction"
    CLASSIFICATION = "classification"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalyzeDocumentRequest(BaseModel):
    documentId: str
    fileUrl: str | None = None
    fileContent: str | None = None
    fileType: str
    language: str | None = None
    documentTypeHint: str | None = None
    fieldsToExtract: list[str] = Field(default_factory=list)
    callbackUrl: str | None = None

    @model_validator(mode="after")
    def require_file_source(self) -> "AnalyzeDocumentRequest":
        if not self.fileUrl and not self.fileContent:
            raise ValueError("Either fileUrl or fileContent must be provided")
        return self


class AnalyzeDocumentResponse(BaseModel):
    jobId: str
    documentId: str
    status: JobStatus


class JobStatusResponse(BaseModel):
    jobId: str
    documentId: str
    status: JobStatus
    message: str | None = None


class FieldValue(BaseModel):
    value: str
    confidence: float


class RegionalText(BaseModel):
    available: bool = False
    note: str = (
        "Region-based text extraction (bounding box + contained text) is planned "
        "for a future release and is not available in the initial version."
    )
    regions: list[Any] = Field(default_factory=list)


class PageResult(BaseModel):
    pageNumber: int
    text: str
    regionalText: RegionalText = Field(default_factory=RegionalText)


class TableResult(BaseModel):
    pageNumber: int
    tableIndex: int = 0
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    html: str | None = None


class JobResultResponse(BaseModel):
    jobId: str
    documentId: str
    status: JobStatus
    documentType: str | None = None
    language: str | None = None
    ocrText: str | None = None
    extractedFields: dict[str, FieldValue] = Field(default_factory=dict)
    tables: list[TableResult] = Field(default_factory=list)
    pages: list[PageResult] = Field(default_factory=list)
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class CallbackNotification(BaseModel):
    jobId: str
    documentId: str
    status: JobStatus
    resultUrl: str | None = None
    documentType: str | None = None
    ocrText: str | None = None
    extractedFields: dict[str, Any] | None = None
    tables: list[Any] | None = None
    error: str | None = None
