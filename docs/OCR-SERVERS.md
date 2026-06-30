# Surya & Chandra OCR Server Guide

Server setup for GPU OCR with web upload UIs.

**Hardware:** 2× NVIDIA RTX 3090, CUDA 12.9 driver  
**Host:** `<your-server-ip>` (replace with your machine's IP or hostname)

---

## Quick reference

| | Surya | Chandra |
|---|---|---|
| **Conda env** | `surya` | `chandra` |
| **App folder** | repo root (`app/`) | `apps/chandra/` |
| **UI URL** | `http://<your-server-ip>:7860` | `http://<your-server-ip>:7861` |
| **vLLM Docker name** | `surya-vllm` | `chandra-vllm` |
| **vLLM host port** | `8001` → container `8000` | `8002` → container `8000` |
| **GPU** | GPU 0 | GPU 1 |
| **HuggingFace model** | `datalab-to/surya-ocr-2` | `datalab-to/chandra-ocr-2` |
| **vLLM served name** | `datalab-to/surya-ocr-2` | `chandra` |

**Important:** Use **separate conda envs**. Surya needs `openai<2`, Chandra needs `openai>=2`.

---

## 1. One-time setup

### Conda environments

```bash
# Surya env
conda create -n surya python=3.10 -y
conda activate surya
pip install -r requirements.txt   # from ocr-surya-app root

# Chandra env (cloned from surya)
conda create -n chandra --clone surya -y
conda activate chandra
pip uninstall -y surya-ocr transformers torch torchvision triton
pip install chandra-ocr gradio
```

### Download models (on host, not in Docker)

```bash
# Fix permissions if Docker ever wrote as root
sudo chown -R "$USER:$USER" ~/.cache/huggingface

conda activate surya
hf download datalab-to/surya-ocr-2

conda activate chandra
hf download datalab-to/chandra-ocr-2
```

### Docker image

```bash
docker pull vllm/vllm-openai:v0.20.1-cu129
```

Use **`v0.20.1-cu129`** (CUDA 12.9). Do **not** use the default `vllm/vllm-openai` tag — it requires CUDA 13.

---

## 2. Start everything (normal workflow)

Run in **3 terminals**, or use tmux (see section 5).

### Terminal 1 — Surya vLLM (GPU 0)

```bash
HF_CACHE="$HOME/.cache/huggingface"
HOST_PORT=8001

docker rm -f surya-vllm 2>/dev/null

docker run -d --name surya-vllm \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  --runtime nvidia --gpus device=0 \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -p ${HOST_PORT}:8000 --ipc=host \
  vllm/vllm-openai:v0.20.1-cu129 \
  --model datalab-to/surya-ocr-2 \
  --no-enforce-eager \
  --max-num-seqs 32 \
  --dtype bfloat16 \
  --max-model-len 18000 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 6291456}' \
  --served-model-name datalab-to/surya-ocr-2 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'

until curl -sf http://127.0.0.1:${HOST_PORT}/health >/dev/null; do echo "waiting surya vLLM..."; sleep 15; done
echo "Surya vLLM ready"
```

### Terminal 2 — Chandra vLLM (GPU 1)

**Stop Ollama first** if it is using GPU 1 (~8 GB VRAM):

```bash
sudo systemctl stop ollama 2>/dev/null || pkill ollama
```

```bash
HF_CACHE="$HOME/.cache/huggingface"
HOST_PORT=8002

docker rm -f chandra-vllm 2>/dev/null

docker run -d --name chandra-vllm \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  --runtime nvidia --gpus device=1 \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -p ${HOST_PORT}:8000 --ipc=host \
  vllm/vllm-openai:v0.20.1-cu129 \
  --model datalab-to/chandra-ocr-2 \
  --no-enforce-eager \
  --max-num-seqs 8 \
  --dtype bfloat16 \
  --max-model-len 12000 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.55 \
  --enable-prefix-caching \
  --mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 6291456}' \
  --served-model-name chandra

until curl -sf http://127.0.0.1:${HOST_PORT}/health >/dev/null; do echo "waiting chandra vLLM..."; sleep 15; done
echo "Chandra vLLM ready"
```

First load: **3–10 minutes** per model.

### Terminal 3 — Surya UI

```bash
cd /path/to/ocr-surya-app
export SURYA_INFERENCE_URL=http://127.0.0.1:8001/v1
./scripts/start.sh
```

### Terminal 4 — Chandra UI

```bash
cd /path/to/ocr-surya-app/apps/chandra
./start.sh
```

Open:
- Surya: `http://<your-server-ip>:7860`
- Chandra: `http://<your-server-ip>:7861`

---

## 3. Port mapping rule

Docker format is always:

```
-p HOST_PORT:8000
```

vLLM listens on **8000 inside the container**. Example: `-p 8002:8000` → health check at `http://127.0.0.1:8002/health`.

### Find a free port

```bash
python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
```

---

## 4. Keep running after closing terminal

### Option A — tmux (simple)

```bash
tmux new -s ocr

# run vLLM + UI commands inside, then detach with: Ctrl+B, then D
tmux attach -t ocr   # reattach later
```

### Option B — systemd (survives reboot)

See separate service files if you set them up. Useful commands:

```bash
sudo systemctl status surya-vllm surya-ui chandra-vllm chandra-ui
sudo systemctl restart surya-ui
```

---

## 5. Nginx (optional — clean URL without :7860)

Surya on port 80:

```bash
sudo tee /etc/nginx/sites-available/surya <<'EOF'
server {
    listen 80;
    server_name <your-server-ip>;
    client_max_body_size 50M;
    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 600s;
    }
}
EOF
sudo ln -sf /etc/nginx/sites-available/surya /etc/nginx/sites-enabled/surya
sudo nginx -t && sudo systemctl reload nginx
```

For Chandra on `/chandra`, add a second `location` block proxying to `7861`.

---

## 6. Troubleshooting

### UI hangs / no output after upload

1. Check vLLM health:
   ```bash
   curl http://127.0.0.1:8001/health   # surya
   curl http://127.0.0.1:8002/health   # chandra
   ```
2. Check Docker logs:
   ```bash
   docker logs surya-vllm --tail 50
   docker logs chandra-vllm --tail 50
   ```
3. First request can take several minutes while the model warms up.

### `waiting...` never finishes

Container probably crashed. Check:

```bash
docker ps -a | grep -E "surya|chandra"
docker logs chandra-vllm 2>&1 | tail -30
```

### Docker cannot reach Hugging Face

Your Docker DNS may not work. Always use offline mode when the model is already downloaded:

```bash
-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
```

### Permission denied on model download

```bash
sudo chown -R "$USER:$USER" ~/.cache/huggingface
```

### Chandra: `max_tokens cannot be greater than max_model_len`

`MAX_OUTPUT_TOKENS` in `apps/chandra/start.sh` must be **less than** vLLM `--max-model-len`. Current setting: `8000` with `max-model-len 12000`.

### Chandra: GPU out of memory

Ollama or other apps may be using GPU 1. Check:

```bash
nvidia-smi
```

Stop Ollama or lower `--gpu-memory-utilization` (e.g. `0.55`).

### Surya env broken after installing Chandra in wrong env

```bash
conda activate surya
pip uninstall -y chandra-ocr markdownify
pip install "openai>=1.55,<2" surya-ocr
```

Never install `chandra-ocr` inside the `surya` env.

### Stale vLLM sentinel (Surya auto-spawn issues)

If using Surya without `SURYA_INFERENCE_URL`:

```bash
rm -f ~/.cache/datalab/surya/vllm_server.json
docker rm -f $(docker ps -aq --filter name=surya-vllm) 2>/dev/null
```

We recommend always starting vLLM manually and setting `SURYA_INFERENCE_URL` (as in `scripts/start.sh`).

---

## 7. Stop everything

```bash
# Stop UIs: Ctrl+C in their terminals

docker stop surya-vllm chandra-vllm
docker rm surya-vllm chandra-vllm
```

---

## 8. File locations

```
/path/to/ocr-surya-app/
├── app/
│   ├── main.py       # FastAPI + Gradio UI
│   ├── api/          # REST API
│   └── services/     # OCR + table extraction
├── scripts/
│   └── start.sh      # activates surya env, sets SURYA_INFERENCE_URL
├── apps/chandra/
│   ├── app.py        # Gradio UI
│   └── start.sh      # activates chandra env, VLLM_API_BASE, MAX_OUTPUT_TOKENS
└── docs/
    └── OCR-SERVERS.md  # this file

~/.cache/huggingface/   # shared model cache (mounted into Docker)
```

---

## 9. Environment variables cheat sheet

### Surya (`scripts/start.sh`)

| Variable | Value |
|---|---|
| `SURYA_INFERENCE_URL` | `http://127.0.0.1:8001/v1` |
| `SURYA_INFERENCE_BACKEND` | `vllm` |
| `VLLM_DOCKER_IMAGE` | `vllm/vllm-openai:v0.20.1-cu129` |

### Chandra (`apps/chandra/start.sh`)

| Variable | Value |
|---|---|
| `VLLM_API_BASE` | `http://127.0.0.1:8002/v1` |
| `VLLM_MODEL_NAME` | `chandra` |
| `MAX_OUTPUT_TOKENS` | `8000` |
