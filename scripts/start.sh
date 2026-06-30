#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate surya

export SURYA_INFERENCE_BACKEND=vllm
export SURYA_INFERENCE_KEEP_ALIVE=1
export SURYA_INFERENCE_URL="${SURYA_INFERENCE_URL:-http://127.0.0.1:8001/v1}"
export VLLM_GPUS="${VLLM_GPUS:-0}"
export VLLM_GPU_TYPE=3090
export VLLM_DTYPE=bfloat16
export VLLM_DOCKER_IMAGE=vllm/vllm-openai:v0.20.1-cu129

export SURYA_AUTH_USER="${SURYA_AUTH_USER:-admin}"
export SURYA_AUTH_PASSWORD="${SURYA_AUTH_PASSWORD:-surya}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
export QWEN_MODEL="${QWEN_MODEL:-qwen2.5:3b}"
export DOC_INTEL_API_TOKEN="${DOC_INTEL_API_TOKEN:-}"
export DOC_INTEL_PUBLIC_BASE_URL="${DOC_INTEL_PUBLIC_BASE_URL:-http://127.0.0.1:7860}"
export SERVER_PORT="${SERVER_PORT:-${SURYA_UI_PORT:-7860}}"

exec python -m app.main
