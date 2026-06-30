import re
from html.parser import HTMLParser
from typing import Any


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._cell_parts: list[str] = []
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._in_cell = True
            self._cell_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            text = " ".join(self._cell_parts).strip()
            text = re.sub(r"\s+", " ", text)
            if self._current_row is not None:
                self._current_row.append(text)
            self._in_cell = False
            self._cell_parts = []
        elif tag == "tr" and self._current_row is not None:
            if any(cell.strip() for cell in self._current_row):
                self.rows.append(self._current_row)
            self._current_row = None

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_parts.append(data.strip())


def parse_html_table(html: str) -> dict[str, Any]:
    parser = _TableHTMLParser()
    try:
        parser.feed(html)
    except Exception:
        return {"headers": [], "rows": [], "html": html}

    rows = parser.rows
    if not rows:
        return {"headers": [], "rows": [], "html": html}

    # First row as headers if any cell looks like a header row (heuristic: short labels)
    if len(rows) > 1 and all(len(c) < 80 for c in rows[0]):
        return {"headers": rows[0], "rows": rows[1:], "html": html}
    return {"headers": [], "rows": rows, "html": html}


def tables_from_predictions(predictions) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    for page_idx, page in enumerate(predictions):
        table_idx = 0
        for block in page.blocks:
            if block.skipped or block.error or not block.html:
                continue
            if block.label != "Table":
                continue
            parsed = parse_html_table(block.html)
            if not parsed["headers"] and not parsed["rows"]:
                # Plain text table block — keep raw content as single-column rows
                text = re.sub(r"<[^>]+>", " ", block.html)
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    parsed["rows"] = [[line] for line in text.split("\n") if line.strip()]
            tables.append(
                {
                    "pageNumber": page_idx + 1,
                    "tableIndex": table_idx,
                    "headers": parsed["headers"],
                    "rows": parsed["rows"],
                    "html": parsed["html"],
                }
            )
            table_idx += 1
    return tables
