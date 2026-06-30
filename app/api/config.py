import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.getenv("DOC_INTEL_DATA_DIR", ROOT_DIR / "data"))
JOBS_DB_PATH = DATA_DIR / "jobs.db"
UPLOADS_DIR = DATA_DIR / "uploads"

SERVICE_VERSION = os.getenv("DOC_INTEL_VERSION", "1.0.0")
API_PREFIX = "/api/v1"
API_BEARER_TOKEN = os.getenv("DOC_INTEL_API_TOKEN", "")

CALLBACK_FULL_RESULT = os.getenv("DOC_INTEL_CALLBACK_FULL_RESULT", "false").lower() in (
    "1",
    "true",
    "yes",
)
PUBLIC_BASE_URL = os.getenv(
    "DOC_INTEL_PUBLIC_BASE_URL",
    f"http://127.0.0.1:{os.getenv('SERVER_PORT', '7860')}",
)
