import base64
import mimetypes
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx
from PIL import Image

from app.api.config import UPLOADS_DIR

SUPPORTED_IMAGE_TYPES = {"png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp", "gif"}


def _save_bytes(job_id: str, data: bytes, ext: str) -> Path:
    path = UPLOADS_DIR / f"{job_id}.{ext.lstrip('.')}"
    path.write_bytes(data)
    return path


def _load_image_bytes(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data)).convert("RGB")


def _pdf_to_images(data: bytes, dpi: int = 200) -> list[Image.Image]:
    import fitz

    images: list[Image.Image] = []
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    finally:
        doc.close()
    return images


def load_document_pages(
    *,
    job_id: str,
    file_url: str | None,
    file_content_b64: str | None,
    file_type: str,
) -> tuple[list[Image.Image], Path | None]:
    file_type = file_type.lower().lstrip(".")
    raw: bytes
    saved: Path | None = None

    if file_content_b64:
        raw = base64.b64decode(file_content_b64)
        saved = _save_bytes(job_id, raw, file_type)
    elif file_url:
        parsed = urlparse(file_url)
        if parsed.scheme in ("http", "https"):
            with httpx.Client(timeout=120.0, follow_redirects=True) as client:
                resp = client.get(file_url)
                resp.raise_for_status()
                raw = resp.content
            ext = file_type or _guess_ext(file_url, resp.headers.get("content-type"))
            saved = _save_bytes(job_id, raw, ext)
        else:
            path = Path(unquote(file_url))
            if not path.is_file():
                raise FileNotFoundError(f"fileUrl not found: {file_url}")
            raw = path.read_bytes()
            saved = path
    else:
        raise ValueError("No file source provided")

    if file_type == "pdf":
        return _pdf_to_images(raw), saved

    if file_type in SUPPORTED_IMAGE_TYPES:
        return [_load_image_bytes(raw)], saved

    raise ValueError(f"Unsupported fileType: {file_type}")


def _guess_ext(url: str, content_type: str | None) -> str:
    path = urlparse(url).path
    if "." in path:
        return path.rsplit(".", 1)[-1].lower()
    if content_type:
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if ext:
            return ext.lstrip(".")
    return "bin"
