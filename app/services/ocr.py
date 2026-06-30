import os
import time
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

os.environ.setdefault("SURYA_INFERENCE_BACKEND", "vllm")
os.environ.setdefault("SURYA_INFERENCE_URL", "http://127.0.0.1:8001/v1")

from surya.inference import SuryaInferenceManager
from surya.recognition import RecognitionPredictor

from app.services.tables import tables_from_predictions

_manager: SuryaInferenceManager | None = None
_predictor: RecognitionPredictor | None = None


@dataclass
class OcrResult:
    page_texts: list[str] = field(default_factory=list)
    full_text: str = ""
    elapsed: float = 0.0
    tables: list[dict[str, Any]] = field(default_factory=list)


def _get_predictor() -> RecognitionPredictor:
    global _manager, _predictor
    if _predictor is None:
        _manager = SuryaInferenceManager()
        _predictor = RecognitionPredictor(_manager)
    return _predictor


def _page_text(page) -> str:
    parts: list[str] = []
    for block in page.blocks:
        if block.skipped or block.error or not block.html:
            continue
        parts.append(f"[{block.label}]\n{block.html}")
    return "\n\n".join(parts)


def run_ocr(images: list[Image.Image]) -> OcrResult:
    if not images:
        return OcrResult()

    start = time.perf_counter()
    predictions = _get_predictor()(images)
    elapsed = time.perf_counter() - start

    page_texts = [_page_text(page) for page in predictions]
    full_text = "\n\n--- page break ---\n\n".join(
        f"Page {i + 1}\n{t}" for i, t in enumerate(page_texts) if t
    )
    tables = tables_from_predictions(predictions)

    return OcrResult(
        page_texts=page_texts,
        full_text=full_text,
        elapsed=elapsed,
        tables=tables,
    )


def run_ocr_single(image: Image.Image) -> tuple[str, float, list[dict[str, Any]]]:
    result = run_ocr([image])
    text = result.page_texts[0] if result.page_texts else result.full_text
    if not text:
        text = "(no text detected)"
    return text, result.elapsed, result.tables
