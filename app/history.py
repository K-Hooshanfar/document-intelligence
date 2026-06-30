import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DOC_INTEL_DATA_DIR", ROOT_DIR / "data"))
DB_PATH = DATA_DIR / "history.db"
IMAGES_DIR = DATA_DIR / "images"


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                created_at TEXT NOT NULL,
                elapsed_seconds REAL NOT NULL,
                image_path TEXT,
                output TEXT NOT NULL,
                document_type TEXT,
                classification_confidence REAL
            )
            """
        )
        for stmt in (
            "ALTER TABLE runs ADD COLUMN document_type TEXT",
            "ALTER TABLE runs ADD COLUMN classification_confidence REAL",
        ):
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError:
                pass
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_username ON runs(username, id DESC)"
        )


def _preview(text: str, limit: int = 120) -> str:
    one_line = " ".join(text.split())
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 3] + "..."


def _resolve_image_path(stored: str | None) -> Path | None:
    if not stored:
        return None
    path = Path(stored)
    if path.is_file():
        return path
    relative = IMAGES_DIR / stored
    if relative.is_file():
        return relative
    by_name = IMAGES_DIR / path.name
    if by_name.is_file():
        return by_name
    return None


def add_run(
    username: str,
    elapsed_seconds: float,
    output: str,
    image: Image.Image | None = None,
    document_type: str | None = None,
    classification_confidence: float | None = None,
) -> int:
    created_at = datetime.now(timezone.utc).isoformat()
    image_filename = None
    if image is not None:
        image_filename = f"{username}_{created_at.replace(':', '-')}.jpg"
        image.convert("RGB").save(IMAGES_DIR / image_filename, format="JPEG", quality=85)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            INSERT INTO runs (
                username, created_at, elapsed_seconds, image_path, output,
                document_type, classification_confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                username,
                created_at,
                elapsed_seconds,
                image_filename,
                output,
                document_type,
                classification_confidence,
            ),
        )
        return int(cursor.lastrowid)


def list_runs(username: str, limit: int = 100) -> list[dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, created_at, elapsed_seconds, output,
                   document_type, classification_confidence
            FROM runs
            WHERE username = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (username, limit),
        ).fetchall()

    return [
        {
            "id": row["id"],
            "created_at": row["created_at"],
            "elapsed_seconds": row["elapsed_seconds"],
            "preview": _preview(row["output"]),
            "document_type": row["document_type"] or "",
            "classification_confidence": row["classification_confidence"],
        }
        for row in rows
    ]


def delete_run(run_id: int, username: str) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT image_path FROM runs WHERE id = ? AND username = ?",
            (run_id, username),
        ).fetchone()
        if row is None:
            return False
        conn.execute(
            "DELETE FROM runs WHERE id = ? AND username = ?",
            (run_id, username),
        )

    path = _resolve_image_path(row["image_path"])
    if path and path.exists():
        path.unlink()
    return True


def get_run(run_id: int, username: str) -> dict[str, Any] | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT id, created_at, elapsed_seconds, image_path, output,
                   document_type, classification_confidence
            FROM runs
            WHERE id = ? AND username = ?
            """,
            (run_id, username),
        ).fetchone()

    if row is None:
        return None

    image_path = _resolve_image_path(row["image_path"])
    image = None
    if image_path:
        image = Image.open(image_path).convert("RGB")

    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "elapsed_seconds": row["elapsed_seconds"],
        "output": row["output"],
        "image": image,
        "image_path": str(image_path) if image_path else None,
        "document_type": row["document_type"] or "",
        "classification_confidence": row["classification_confidence"],
    }
