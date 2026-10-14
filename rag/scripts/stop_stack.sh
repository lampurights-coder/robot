#!/usr/bin/env bash
# Stop FAQ API + llama. Qdrant stays up unless STOP_QDRANT=1.
set -euo pipefail

RAG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLLM="$(cd "$RAG/../.." && pwd)"   # ollm-ocr-2
PIDFILE_API="${PIDFILE_API:-$RAG/fastapi_faq.pid}"

echo "Stopping FAQ API..."
if [[ -f "$PIDFILE_API" ]]; then
  kill "$(cat "$PIDFILE_API")" 2>/dev/null || true
  rm -f "$PIDFILE_API"
fi
pkill -f 'scripts/fastapi_faq_app.py' 2>/dev/null || true
pkill -f 'scripts/gradio_faq_app.py' 2>/dev/null || true

echo "Stopping llama-server..."
if [[ -x "$OLLM/scripts/qwen_llama_stop.sh" ]]; then
  "$OLLM/scripts/qwen_llama_stop.sh" || true
fi
pkill -f 'llama-server' 2>/dev/null || true

if [[ "${STOP_QDRANT:-0}" == "1" ]]; then
  echo "Stopping Qdrant docker..."
  docker stop qdrant-tita 2>/dev/null || true
fi

sleep 1
echo "Done."
ss -tlnp 2>/dev/null | grep -E ':6333|:7860|:8080' || echo "7860/8080 free"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null || true
