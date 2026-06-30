# Surya Document Intelligence

GPU-accelerated document OCR and analysis built on [Surya OCR](https://github.com/datalab-to/surya). Upload images or PDFs, extract text and tables, classify document type, and pull structured fields — via a Gradio web UI or a REST API.

## Features

- **OCR** — Surya OCR 2 via vLLM (layout-aware text extraction)
- **Table extraction** — structured tables from OCR layout output
- **Document classification** — invoice, letter, contract, receipt, and more (Qwen via Ollama)
- **Field extraction** — user-defined fields from OCR text (multilingual labels supported)
- **Web UI** — Gradio app with Process and History tabs, basic auth
- **REST API** — async jobs with status polling and optional webhooks
- **Docker** — vLLM inference + UI/API in one `docker compose` stack

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Gradio UI  │────▶│  FastAPI     │────▶│  Surya (vLLM)   │
│  + History  │     │  + Worker    │     │  GPU inference  │
└─────────────┘     └──────┬───────┘     └─────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Ollama/Qwen  │
                    │ classify +   │
                    │ extract      │
                    └──────────────┘
```

| Component | Default port | Description |
|-----------|--------------|-------------|
| UI + API | `7860` | Gradio UI, `/health`, `/api/v1/*` |
| vLLM | `8001` | OpenAI-compatible Surya backend |

## Requirements

- **GPU** — NVIDIA GPU with CUDA 12.x (tested on RTX 3090)
- **Docker** — NVIDIA Container Toolkit for GPU passthrough
- **Model** — `datalab-to/surya-ocr-2` on Hugging Face (download before offline use)
- **Ollama** (optional) — for classification and field extraction (`qwen2.5:3b` by default)

## Quick start (Docker)

1. Copy environment file and edit secrets:

```bash
cp .env.example .env
# Set SURYA_AUTH_PASSWORD and DOC_INTEL_API_TOKEN
```

2. Download the Surya model (on the host, into your Hugging Face cache):

```bash
pip install huggingface-hub
hf download datalab-to/surya-ocr-2
```

3. Start the stack:

```bash
./scripts/docker-up.sh
# or: docker compose up -d
```

4. Open the UI at `http://<host>:7860` (login with `SURYA_AUTH_USER` / `SURYA_AUTH_PASSWORD`).

Use `docker compose logs -f` to watch startup. The vLLM container can take several minutes on first boot.

## Local development (conda)

For running the UI on the host while vLLM runs in Docker:

```bash
conda create -n surya python=3.10 -y
conda activate surya
pip install -r requirements.txt

# Start vLLM (see docker-compose.yml or docs/OCR-SERVERS.md)
export SURYA_INFERENCE_BACKEND=vllm
export SURYA_INFERENCE_URL=http://127.0.0.1:8001/v1
./scripts/start.sh
```

See [docs/OCR-SERVERS.md](./docs/OCR-SERVERS.md) for detailed GPU server setup, including a separate [Chandra OCR](./apps/chandra/) UI on port `7861`.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SURYA_AUTH_USER` | `admin` | Gradio login username |
| `SURYA_AUTH_PASSWORD` | — | Gradio login password |
| `VLLM_GPU` | `0` | GPU index for vLLM container |
| `VLLM_HOST_PORT` | `8001` | Host port mapped to vLLM |
| `UI_HOST_PORT` | `7860` | Host port for UI/API |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama API URL |
| `QWEN_MODEL` | `qwen2.5:3b` | Model for classification / fields |
| `DOC_INTEL_API_TOKEN` | — | Bearer token for REST API |
| `DOC_INTEL_PUBLIC_BASE_URL` | — | Public base URL for API callbacks |

## REST API

All API routes require `Authorization: Bearer <DOC_INTEL_API_TOKEN>`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health check |
| `POST` | `/api/v1/documents/analyze` | Submit document for async processing |
| `GET` | `/api/v1/jobs/{jobId}` | Job status |
| `GET` | `/api/v1/jobs/{jobId}/result` | Job result (when completed) |

Example analyze request:

```bash
curl -X POST "http://localhost:7860/api/v1/documents/analyze" \
  -H "Authorization: Bearer $DOC_INTEL_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "documentId": "doc-001",
    "fileContent": "<base64-encoded-image>",
    "fileType": "image/png",
    "documentTypeHint": "invoice",
    "fieldsToExtract": ["date", "total", "vendor"]
  }'
```

## Project layout

```
├── app/
│   ├── main.py           # FastAPI + Gradio entrypoint
│   ├── ui.py             # Gradio UI
│   ├── classifier.py     # Ollama/Qwen classification & field extraction
│   ├── history.py        # SQLite run history
│   ├── api/              # REST API routes, worker, schemas
│   └── services/
│       ├── ocr.py        # OCR wrapper
│       └── tables.py     # Table extraction
├── apps/
│   └── chandra/          # Optional Chandra OCR UI (separate stack)
├── scripts/
│   ├── start.sh          # Run app locally
│   ├── docker-up.sh
│   └── docker-down.sh
├── docs/
│   └── OCR-SERVERS.md    # Ops guide for dual-GPU deployment
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Related projects

- [Surya OCR](https://github.com/datalab-to/surya) — OCR engine (`surya-ocr` on PyPI)
- [Chandra OCR](https://huggingface.co/datalab-to/chandra-ocr-2) — alternative OCR model (see `apps/chandra/`)
