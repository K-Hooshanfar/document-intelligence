#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chandra
export VLLM_API_BASE="${VLLM_API_BASE:-http://127.0.0.1:8002/v1}"
export VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-chandra}"
export MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-8000}"
exec python app.py
