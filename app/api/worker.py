import asyncio
import logging
from typing import Any

import httpx

from app import classifier
from app.api import jobs_db
from app.api.config import CALLBACK_FULL_RESULT, PUBLIC_BASE_URL
from app.api.document_loader import load_document_pages
from app.api.schemas import (
    AnalyzeDocumentRequest,
    CallbackNotification,
    FieldValue,
    JobResultResponse,
    JobStatus,
    PageResult,
    RegionalText,
    TableResult,
)
from app.services.ocr import run_ocr

logger = logging.getLogger(__name__)

_queue: asyncio.Queue[str] = asyncio.Queue()
_worker_task: asyncio.Task | None = None


async def start_worker() -> None:
    global _worker_task
    if _worker_task is None:
        _worker_task = asyncio.create_task(_worker_loop())


async def stop_worker() -> None:
    global _worker_task
    if _worker_task is not None:
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
        _worker_task = None


async def enqueue(job_id: str) -> None:
    await _queue.put(job_id)


async def _worker_loop() -> None:
    while True:
        job_id = await _queue.get()
        try:
            await asyncio.to_thread(_process_job, job_id)
        except Exception:
            logger.exception("Unhandled error processing job %s", job_id)
            job = jobs_db.get_job(job_id)
            jobs_db.update_job(
                job_id,
                status=JobStatus.FAILED,
                message="Internal processing error",
                result=JobResultResponse(
                    jobId=job_id,
                    documentId=job["document_id"] if job else "",
                    status=JobStatus.FAILED,
                    error="Internal processing error",
                ),
            )
        finally:
            _queue.task_done()


def _merge_tables(
    ocr_tables: list[dict[str, Any]],
    qwen_tables: list[dict[str, Any]],
) -> list[TableResult]:
    seen_html = {t.get("html", "") for t in ocr_tables if t.get("html")}
    merged: list[TableResult] = []

    for t in ocr_tables:
        merged.append(TableResult.model_validate(t))

    for t in qwen_tables:
        html = t.get("html", "")
        if html and html in seen_html:
            continue
        if not t.get("rows") and not t.get("headers"):
            continue
        merged.append(TableResult.model_validate(t))

    return merged


def _process_job(job_id: str) -> None:
    job = jobs_db.get_job(job_id)
    if job is None:
        return

    request = AnalyzeDocumentRequest.model_validate(job["request"])
    document_id = request.documentId

    try:
        jobs_db.update_job(
            job_id,
            status=JobStatus.OCR_PROCESSING,
            message="Running OCR",
        )

        images, _ = load_document_pages(
            job_id=job_id,
            file_url=request.fileUrl,
            file_content_b64=request.fileContent,
            file_type=request.fileType,
        )
        ocr_result = run_ocr(images)

        pages = [
            PageResult(
                pageNumber=i + 1,
                text=text or "",
                regionalText=RegionalText(),
            )
            for i, text in enumerate(ocr_result.page_texts)
        ]

        jobs_db.update_job(
            job_id,
            status=JobStatus.CLASSIFICATION,
            message="Classifying document type",
        )
        document_type, _ = classifier.classify_document(
            ocr_result.full_text,
            hint=request.documentTypeHint,
            language=request.language,
        )

        extracted: dict[str, FieldValue] = {}
        if request.fieldsToExtract:
            jobs_db.update_job(
                job_id,
                status=JobStatus.FIELD_EXTRACTION,
                message="Extracting fields",
            )
            raw_fields = classifier.extract_fields(
                ocr_result.full_text,
                request.fieldsToExtract,
                document_type=document_type,
                language=request.language,
            )
            extracted = {
                name: FieldValue(
                    value=str(item["value"]),
                    confidence=float(item["confidence"]),
                )
                for name, item in raw_fields.items()
            }

        jobs_db.update_job(
            job_id,
            status=JobStatus.TABLE_EXTRACTION,
            message="Extracting tables",
        )
        qwen_tables: list[dict[str, Any]] = []
        if not ocr_result.tables:
            try:
                qwen_tables = classifier.extract_tables_from_text(ocr_result.full_text)
            except Exception:
                logger.exception("Qwen table extraction failed for job %s", job_id)

        tables = _merge_tables(ocr_result.tables, qwen_tables)

        result = JobResultResponse(
            jobId=job_id,
            documentId=document_id,
            status=JobStatus.COMPLETED,
            documentType=document_type,
            language=request.language,
            ocrText=ocr_result.full_text,
            extractedFields=extracted,
            tables=tables,
            pages=pages,
        )
        jobs_db.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            message="Processing completed",
            result=result,
        )

        if request.callbackUrl:
            _send_callback(request, result)

    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        result = JobResultResponse(
            jobId=job_id,
            documentId=document_id,
            status=JobStatus.FAILED,
            error=str(exc),
        )
        jobs_db.update_job(
            job_id,
            status=JobStatus.FAILED,
            message=str(exc),
            result=result,
        )
        if request.callbackUrl:
            _send_callback(request, result)


def _send_callback(request: AnalyzeDocumentRequest, result: JobResultResponse) -> None:
    if CALLBACK_FULL_RESULT and result.status == JobStatus.COMPLETED:
        payload = CallbackNotification(
            jobId=result.jobId,
            documentId=result.documentId,
            status=result.status,
            documentType=result.documentType,
            ocrText=result.ocrText,
            extractedFields={
                k: v.value for k, v in (result.extractedFields or {}).items()
            },
            tables=[t.model_dump() for t in result.tables],
        )
    else:
        payload = CallbackNotification(
            jobId=result.jobId,
            documentId=result.documentId,
            status=result.status,
            resultUrl=f"{PUBLIC_BASE_URL}/api/v1/jobs/{result.jobId}/result",
            error=result.error,
        )

    try:
        with httpx.Client(timeout=30.0) as client:
            client.post(
                request.callbackUrl,
                json=payload.model_dump(exclude_none=True),
            )
    except Exception:
        logger.exception("Callback failed for job %s", result.jobId)
