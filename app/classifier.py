import json
import os
import re
from typing import Any

import httpx

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen2.5:3b")
MAX_OCR_CHARS = int(os.getenv("MAX_OCR_CHARS_FOR_LLM", "12000"))
SUMMARY_TEMPERATURE = float(os.getenv("SUMMARY_TEMPERATURE", "0.3"))

DOCUMENT_TYPES = [
    "invoice",
    "letter",
    "contract",
    "receipt",
    "form",
    "report",
    "memo",
    "purchase_order",
    "bank_statement",
    "id_document",
    "other",
]


def _truncate(text: str) -> str:
    if len(text) <= MAX_OCR_CHARS:
        return text
    return text[: MAX_OCR_CHARS - 20] + "\n...[truncated]"


def _parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Model did not return JSON: {text[:200]}")
    return json.loads(text[start : end + 1])


def classify_document(
    ocr_text: str,
    *,
    hint: str | None = None,
    language: str | None = None,
) -> tuple[str, float]:
    types_list = ", ".join(DOCUMENT_TYPES)
    hint_line = f"\nSuggested type: {hint}." if hint else ""
    lang_line = f"\nDocument language: {language}." if language else ""

    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a document classification assistant. "
                    "Read the OCR text and classify the document. "
                    'Respond with JSON only: {"documentType": "<type>", "confidence": 0.0-1.0}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Allowed types: {types_list}.{hint_line}{lang_line}\n\n"
                    f"OCR text:\n{_truncate(ocr_text)}"
                ),
            },
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }

    with httpx.Client(timeout=180.0) as client:
        resp = client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        resp.raise_for_status()
        raw = resp.json()["message"]["content"].strip()

    data = _parse_json_object(raw)
    doc_type = str(data.get("documentType", "other")).lower().replace(" ", "_")
    confidence = float(data.get("confidence", 0.8))
    return doc_type, min(max(confidence, 0.0), 1.0)


def _ollama_chat(system: str, user: str, temperature: float = 0.1) -> str:
    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    with httpx.Client(timeout=180.0) as client:
        resp = client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


# Optional English API keys → extra Persian words to search in OCR (not required to know these)
FIELD_SEARCH_HINTS: dict[str, list[str]] = {
    "date": ["تاریخ"],
    "postal_code": ["کد پستی", "کدپستی"],
    "letter_number": ["شماره", "شماره نامه"],
    "invoice_number": ["شماره", "شماره فاکتور"],
    "amount": ["مبلغ", "مبلغ قابل پرداخت"],
    "sender": ["فرستنده"],
    "receiver": ["گیرنده"],
}

# Common spelling variants (no space ↔ with space)
SPACED_VARIANTS: dict[str, str] = {
    "کدپستی": "کد پستی",
    "شمارهنامه": "شماره نامه",
    "شمارهفاکتور": "شماره فاکتور",
}


def parse_field_list(raw: str) -> list[str]:
    """Return field labels exactly as the user typed them."""
    if not raw.strip():
        return []
    parts = re.split(r"[,،;|\n]+", raw)
    seen: set[str] = set()
    result: list[str] = []
    for part in parts:
        label = part.strip()
        if label and label not in seen:
            seen.add(label)
            result.append(label)
    return result


def _label_search_variants(label: str) -> list[str]:
    """All label spellings to search for in OCR text."""
    variants: list[str] = [label]
    collapsed = re.sub(r"\s+", "", label)
    if collapsed != label:
        variants.append(collapsed)
    if collapsed in SPACED_VARIANTS:
        variants.append(SPACED_VARIANTS[collapsed])
    if label in SPACED_VARIANTS.values():
        for tight, spaced in SPACED_VARIANTS.items():
            if spaced == label:
                variants.append(tight)
    # English API key → also search Persian on document
    key = label.lower().replace(" ", "_") if label.isascii() else ""
    if key in FIELD_SEARCH_HINTS:
        variants.extend(FIELD_SEARCH_HINTS[key])
    return list(dict.fromkeys(v for v in variants if v))


def _strip_html(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</t[dh]>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"[ \t]+", " ", text)


_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _normalize_digits(value: str) -> str:
    return value.translate(_PERSIAN_DIGITS).translate(_ARABIC_DIGITS)


def _lookup_field_entry(fields_data: dict, label: str) -> dict | str | None:
    if label in fields_data:
        return fields_data[label]
    for variant in _label_search_variants(label):
        if variant in fields_data:
            return fields_data[variant]
    for key, value in fields_data.items():
        if re.sub(r"\s+", "", str(key)) == re.sub(r"\s+", "", label):
            return value
    return None


def _field_descriptions(labels: list[str]) -> str:
    lines = []
    for label in labels:
        lines.append(
            f'- "{label}": find the value printed after this label in the document '
            f'(often after ":" or "："). Use key "{label}" exactly in JSON.'
        )
    return "\n".join(lines)


def extract_fields(
    ocr_text: str,
    fields: list[str],
    *,
    document_type: str | None = None,
    language: str | None = None,
) -> dict[str, dict[str, float | str]]:
    if not fields:
        return {}

    labels = [f.strip() for f in fields if f.strip()]
    if not labels:
        return {}

    clean_text = _strip_html(ocr_text)
    doc_ctx = f" Document type: {document_type}." if document_type else ""
    lang_ctx = f" Language: {language or 'fa-en'}."

    system = (
        "You extract fields from OCR text of Persian/English documents. "
        "Each requested name is a label printed on the document (e.g. تاریخ, شماره, تخفیف). "
        "Find the value that appears after that label (usually after ':' or '：'). "
        "The OCR may contain HTML tags — ignore tags and read visible text only. "
        "Only use values that literally appear in the OCR — never invent data. "
        "If a label is not found, omit it from the response. "
        'Return JSON only: {"fields": {"<exact label>": {"value": "...", "confidence": 0.0-1.0}}}'
    )
    user = (
        f"Extract these labels:{doc_ctx}{lang_ctx}\n"
        f"{_field_descriptions(labels)}\n\n"
        f"OCR text:\n{_truncate(clean_text)}"
    )

    raw = _ollama_chat(system, user)
    data = _parse_json_object(raw)
    fields_data = data.get("fields", data)

    result: dict[str, dict[str, float | str]] = {}
    for label in labels:
        entry = _lookup_field_entry(fields_data, label)
        if isinstance(entry, dict):
            value = str(entry.get("value", "")).strip()
            confidence = float(entry.get("confidence", 0.0))
        elif entry is not None:
            value = str(entry).strip()
            confidence = 0.5 if value else 0.0
        else:
            continue

        if value:
            result[label] = {
                "value": _normalize_digits(value),
                "confidence": min(max(confidence, 0.0), 1.0),
            }

    return result


def summarize_document(
    ocr_text: str,
    *,
    document_type: str | None = None,
    language: str | None = None,
) -> str:
    clean_text = _strip_html(ocr_text).strip()
    if not clean_text:
        return ""

    doc_ctx = f" The document type is {document_type}." if document_type else ""
    lang_ctx = (
        f" Write the summary in {language}."
        if language
        else " Write the summary in the same language as the document."
    )

    system = (
        "You summarize documents from their OCR text. "
        "Produce a concise, faithful summary that captures the purpose, "
        "key facts, parties, dates, and amounts mentioned. "
        "The OCR may contain HTML tags — ignore tags and read visible text only. "
        "Only use information present in the text — never invent details. "
        "Return the summary as plain prose, no preamble or headings."
    )
    user = (
        f"Summarize the following document.{doc_ctx}{lang_ctx}\n\n"
        f"OCR text:\n{_truncate(clean_text)}"
    )

    return _ollama_chat(system, user, temperature=SUMMARY_TEMPERATURE).strip()


def extract_tables_from_text(ocr_text: str) -> list[dict[str, Any]]:
    system = (
        "You extract tables from OCR text. "
        "Return JSON only: "
        '{"tables": [{"pageNumber": 1, "tableIndex": 0, "headers": ["col1"], '
        '"rows": [["val1", "val2"]]}]}. '
        "Use empty tables array if no tables exist. "
        "Only include data that appears in the OCR text."
    )
    user = f"OCR text:\n{_truncate(ocr_text)}"

    raw = _ollama_chat(system, user)
    data = _parse_json_object(raw)
    tables = data.get("tables", [])
    if not isinstance(tables, list):
        return []
    return [t for t in tables if isinstance(t, dict)]
