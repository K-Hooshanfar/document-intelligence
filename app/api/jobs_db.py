import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from app.api.config import DATA_DIR, JOBS_DB_PATH, UPLOADS_DIR
from app.api.schemas import AnalyzeDocumentRequest, JobResultResponse, JobStatus


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(JOBS_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                request_json TEXT NOT NULL,
                result_json TEXT,
                file_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_document ON jobs(document_id)"
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_job_id() -> str:
    return f"JOB-{uuid.uuid4().hex[:8].upper()}"


def create_job(request: AnalyzeDocumentRequest) -> str:
    job_id = new_job_id()
    now = _now()
    with sqlite3.connect(JOBS_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO jobs (job_id, document_id, status, message, request_json,
                              result_json, file_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, ?)
            """,
            (
                job_id,
                request.documentId,
                JobStatus.QUEUED.value,
                "Job queued",
                request.model_dump_json(),
                now,
                now,
            ),
        )
    return job_id


def update_job(
    job_id: str,
    *,
    status: JobStatus | None = None,
    message: str | None = None,
    result: JobResultResponse | None = None,
) -> None:
    fields: list[str] = ["updated_at = ?"]
    values: list[Any] = [_now()]

    if status is not None:
        fields.append("status = ?")
        values.append(status.value)
    if message is not None:
        fields.append("message = ?")
        values.append(message)
    if result is not None:
        fields.append("result_json = ?")
        values.append(result.model_dump_json())

    values.append(job_id)
    with sqlite3.connect(JOBS_DB_PATH) as conn:
        conn.execute(
            f"UPDATE jobs SET {', '.join(fields)} WHERE job_id = ?",
            values,
        )


def get_job(job_id: str) -> dict[str, Any] | None:
    with sqlite3.connect(JOBS_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
    if row is None:
        return None
    data = dict(row)
    data["request"] = json.loads(data["request_json"])
    data["result"] = json.loads(data["result_json"]) if data["result_json"] else None
    return data
