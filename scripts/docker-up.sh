#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  echo "Creating .env from .env.example — edit SURYA_AUTH_PASSWORD before exposing this on a network."
  cp .env.example .env
fi

mkdir -p data

echo "Starting Document Intelligence (vLLM + API + UI). First start may take several minutes."
docker compose build surya-ui
docker compose up -d

set -a
# shellcheck disable=SC1091
source .env
set +a

IP="$(hostname -I | awk '{print $1}')"
echo ""
echo "UI:      http://${IP}:${UI_HOST_PORT:-7860}/"
echo "API:     http://${IP}:${UI_HOST_PORT:-7860}/health"
echo "         http://${IP}:${UI_HOST_PORT:-7860}/api/v1/documents/analyze"
echo ""
echo "You should see tabs: Process | History  (not the old 'OCR' tab)."
echo "Process tab shows: Document type, Classification confidence, Extracted fields."
echo "Logs:    docker compose logs -f"
echo "Stop:    ./scripts/docker-down.sh"
