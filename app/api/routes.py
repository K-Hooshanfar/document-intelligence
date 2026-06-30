from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status

from app.api import jobs_db, worker
from app.api.auth import verify_token
from app.api.config import API_PREFIX, SERVICE_VERSION
from app.api.schemas import (
    AnalyzeDocumentRequest,
    AnalyzeDocumentResponse,
    HealthResponse,
    JobResultResponse,
    JobStatus,
    JobStatusResponse,
)


@asynccontextmanager
async def api_lifespan(app: FastAPI):
    jobs_db.init_db()
    await worker.start_worker()
    yield
    await worker.stop_worker()


def create_api_app() -> FastAPI:
    app = FastAPI(
        title="AI Document Intelligence",
        version=SERVICE_VERSION,
        lifespan=api_lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service="AI Document Intelligence",
            version=SERVICE_VERSION,
        )

    @app.post(
        f"{API_PREFIX}/documents/analyze",
        response_model=AnalyzeDocumentResponse,
        dependencies=[Depends(verify_token)],
    )
    async def analyze_document(
        body: AnalyzeDocumentRequest,
    ) -> AnalyzeDocumentResponse:
        job_id = jobs_db.create_job(body)
        await worker.enqueue(job_id)
        return AnalyzeDocumentResponse(
            jobId=job_id,
            documentId=body.documentId,
            status=JobStatus.QUEUED,
        )

    @app.get(
        f"{API_PREFIX}/jobs/{{job_id}}",
        response_model=JobStatusResponse,
        dependencies=[Depends(verify_token)],
    )
    def get_job_status(job_id: str) -> JobStatusResponse:
        job = jobs_db.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return JobStatusResponse(
            jobId=job_id,
            documentId=job["document_id"],
            status=JobStatus(job["status"]),
            message=job.get("message"),
        )

    @app.get(
        f"{API_PREFIX}/jobs/{{job_id}}/result",
        response_model=JobResultResponse,
        dependencies=[Depends(verify_token)],
    )
    def get_job_result(job_id: str) -> JobResultResponse:
        job = jobs_db.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        status_val = JobStatus(job["status"])
        if status_val == JobStatus.FAILED and job.get("result"):
            return JobResultResponse.model_validate(job["result"])

        if status_val != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Job not completed (status: {status_val.value})",
            )

        if not job.get("result"):
            raise HTTPException(status_code=404, detail="Result not available")

        return JobResultResponse.model_validate(job["result"])

    return app
