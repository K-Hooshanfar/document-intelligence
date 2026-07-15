import json
import os

import gradio as gr
from PIL import Image

from app import classifier, history
from app.services.ocr import run_ocr_single

history.init_db()


def _username(request: gr.Request) -> str:
    return request.username or "anonymous"


def _classify(ocr_text: str, hint: str | None) -> tuple[str, str]:
    if not ocr_text or ocr_text == "(no text detected)":
        return "", ""
    try:
        doc_type, confidence = classifier.classify_document(ocr_text, hint=hint or None)
        return doc_type.replace("_", " ").title(), f"{confidence * 100:.1f}%"
    except Exception as exc:
        return "Classification failed", str(exc)


def _history_rows(request: gr.Request) -> list[list]:
    runs = history.list_runs(_username(request))
    return [
        [
            run["id"],
            run["created_at"],
            f"{run['elapsed_seconds']:.2f}s",
            run["document_type"] or "—",
            run["preview"],
        ]
        for run in runs
    ]


def _table_run_id(table: list[list] | None, row_idx: int) -> int:
    if table is None:
        raise ValueError("empty table")
    if hasattr(table, "iloc"):
        return int(table.iloc[row_idx, 0])
    return int(table[row_idx][0])


def _parse_fields(raw: str) -> list[str]:
    return classifier.parse_field_list(raw)


def _format_fields(fields: dict) -> str:
    if not fields:
        return "(no fields found in OCR text)"
    return json.dumps(fields, indent=2, ensure_ascii=False)


def _format_tables(tables: list) -> str:
    if not tables:
        return "(no tables found)"
    return json.dumps(tables, indent=2, ensure_ascii=False)


def ocr_image(
    image: Image.Image | None,
    type_hint: str,
    fields_raw: str,
    request: gr.Request,
) -> tuple[str, str, str, str, str, str, str, list[list]]:
    empty = ("", "", "", "", "(no summary)", "(no fields found in OCR text)", "(no tables found)")
    if image is None:
        return (
            *empty[:4],
            empty[4],
            empty[5],
            empty[6],
            _history_rows(request),
        )

    output, elapsed, tables = run_ocr_single(image)
    timing = f"{elapsed:.2f} seconds"
    doc_type, confidence = _classify(output, type_hint.strip() or None)

    summary = ""
    if output and output != "(no text detected)":
        try:
            summary = classifier.summarize_document(
                output,
                document_type=doc_type or None,
            )
        except Exception as exc:
            summary = f"Summary failed: {exc}"

    field_names = _parse_fields(fields_raw)
    extracted: dict = {}
    if field_names:
        try:
            extracted = classifier.extract_fields(
                output,
                field_names,
                document_type=doc_type or None,
            )
        except Exception as exc:
            extracted = {"error": str(exc)}

    if not tables:
        try:
            tables = classifier.extract_tables_from_text(output)
        except Exception:
            pass

    conf_value = None
    if confidence.endswith("%"):
        try:
            conf_value = float(confidence.rstrip("%")) / 100.0
        except ValueError:
            conf_value = None
    history.add_run(
        _username(request),
        elapsed,
        output,
        image,
        document_type=doc_type or None,
        classification_confidence=conf_value,
    )

    return (
        output,
        timing,
        doc_type,
        confidence,
        summary or "(no summary)",
        _format_fields(extracted),
        _format_tables(tables),
        _history_rows(request),
    )


def refresh_history(request: gr.Request) -> list[list]:
    return _history_rows(request)


def load_history_item(
    run_id: int | str | float | None,
    request: gr.Request,
) -> tuple[Image.Image | None, str, str, str, str]:
    if run_id is None or run_id == "":
        return None, "", "", "", ""

    run = history.get_run(int(run_id), _username(request))
    if run is None:
        return None, "", "Run not found.", "", ""

    timing = f"{run['elapsed_seconds']:.2f} seconds"
    confidence = ""
    if run["classification_confidence"] is not None:
        confidence = f"{run['classification_confidence'] * 100:.1f}%"

    return (
        run["image"],
        run["output"],
        timing,
        run["document_type"] or "—",
        confidence,
    )


def select_history_row(
    evt: gr.SelectData,
    table: list[list] | None,
    request: gr.Request,
) -> tuple[int, Image.Image | None, str, str, str, str]:
    if not table or evt.index[0] is None:
        return 0, None, "", "", "", ""

    run_id = _table_run_id(table, evt.index[0])
    image, output, timing, doc_type, confidence = load_history_item(run_id, request)
    return run_id, image, output, timing, doc_type, confidence


def delete_history_item(
    run_id: int | str | float | None,
    request: gr.Request,
) -> tuple[list[list], int | None, Image.Image | None, str, str, str, str, str]:
    message = ""
    if run_id is None or run_id == "":
        message = "Select a run from the dropdown or click a table row first."
    else:
        deleted = history.delete_run(int(run_id), _username(request))
        message = "Deleted." if deleted else "Run not found."

    return (
        _history_rows(request),
        None,
        None,
        "",
        "",
        "",
        "",
        message,
    )


def _auth() -> list[tuple[str, str]]:
    user = os.getenv("SURYA_AUTH_USER", "admin")
    password = os.getenv("SURYA_AUTH_PASSWORD", "surya")
    return [(user, password)]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Document Intelligence") as demo:
        gr.Markdown(
            "# Document Intelligence\n"
            "Upload a document image for text extraction and automatic document classification."
        )

        with gr.Tabs():
            with gr.Tab("Process"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(type="pil", label="Upload image")
                        type_hint = gr.Textbox(
                            label="Document type hint (optional)",
                            placeholder="e.g. invoice, letter, contract",
                        )
                        fields_input = gr.Textbox(
                            label="Fields to extract — type labels as printed on the document",
                            placeholder="تاریخ، کد پستی، شماره، مبلغ",
                            lines=2,
                        )
                        run_btn = gr.Button("Process document", variant="primary")
                    with gr.Column():
                        doc_type_out = gr.Textbox(label="Document type (classification)")
                        confidence_out = gr.Textbox(label="Classification confidence")
                        summary_out = gr.Textbox(
                            label="Document summary",
                            lines=6,
                        )
                        fields_out = gr.Textbox(
                            label="Extracted fields",
                            lines=8,
                        )
                        tables_out = gr.Textbox(
                            label="Extracted tables",
                            lines=8,
                        )
                        output_text = gr.Textbox(label="Extracted text", lines=12)
                        timing_text = gr.Textbox(label="Processing time")

            with gr.Tab("History"):
                gr.Markdown(
                    "**To view or delete:** click a row in the table, "
                    "or pick a run from the dropdown, then press Delete."
                )
                history_table = gr.Dataframe(
                    headers=["ID", "Time (UTC)", "Duration", "Type", "Preview"],
                    datatype=["number", "str", "str", "str", "str"],
                    interactive=False,
                    label="Your past runs (newest first)",
                )
                history_select = gr.Dropdown(
                    label="Selected run",
                    choices=[],
                    value=None,
                    interactive=True,
                )
                delete_status = gr.Textbox(label="Status", interactive=False)
                with gr.Row():
                    refresh_btn = gr.Button("Refresh")
                    delete_btn = gr.Button("Delete selected", variant="stop")
                with gr.Row():
                    history_image = gr.Image(type="pil", label="Saved image")
                    history_output = gr.Textbox(label="Full output", lines=16)
                with gr.Row():
                    history_doc_type = gr.Textbox(label="Document type")
                    history_confidence = gr.Textbox(label="Classification confidence")
                history_timing = gr.Textbox(label="Processing time")

        run_btn.click(
            ocr_image,
            inputs=[image_input, type_hint, fields_input],
            outputs=[
                output_text,
                timing_text,
                doc_type_out,
                confidence_out,
                summary_out,
                fields_out,
                tables_out,
                history_table,
            ],
        )
        refresh_btn.click(refresh_history, outputs=history_table)

        def _update_dropdown(request: gr.Request) -> gr.Dropdown:
            runs = history.list_runs(_username(request))
            choices = [(f"#{r['id']} — {r['document_type'] or 'unknown'}", r["id"]) for r in runs]
            return gr.Dropdown(choices=choices, value=None, label="Selected run")

        demo.load(_update_dropdown, outputs=history_select)
        demo.load(refresh_history, outputs=history_table)

        history_select.change(
            load_history_item,
            inputs=[history_select],
            outputs=[
                history_image,
                history_output,
                history_timing,
                history_doc_type,
                history_confidence,
            ],
        )
        history_table.select(
            select_history_row,
            inputs=[history_table],
            outputs=[
                history_select,
                history_image,
                history_output,
                history_timing,
                history_doc_type,
                history_confidence,
            ],
        )
        delete_btn.click(
            delete_history_item,
            inputs=[history_select],
            outputs=[
                history_table,
                history_select,
                history_image,
                history_output,
                history_timing,
                history_doc_type,
                history_confidence,
                delete_status,
            ],
        ).then(_update_dropdown, outputs=history_select)
        refresh_btn.click(_update_dropdown, outputs=history_select)

    return demo
