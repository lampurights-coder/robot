#!/usr/bin/env bash
# Start: Qdrant (vector DB) → optional llama → FAQ FastAPI (Jina embed + BGE rerank + weather tool)
#
# Usage:
#   ./scripts/start_stack.sh              # qdrant + API (cuda embed/rerank)
#   START_LLM=1 ./scripts/start_stack.sh  # also start Qwen on :8080
#   EMBED_DEVICE=cpu ./scripts/start_stack.sh
#
# Stop:
#   ./scripts/stop_stack.sh

set -euo pipefail

RAG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$RAG/.." && pwd)"          # tita-arshia
OLLM="$(cd "$ROOT/.." && pwd)"         # ollm-ocr-2
AGENT="${WEATHER_AGENT_ROOT:-$ROOT/agent}"
LOG_DIR="${LOG_DIR:-$RAG/logs}"
PIDFILE_API="${PIDFILE_API:-$RAG/fastapi_faq.pid}"
LOG_API="${LOG_API:-$LOG_DIR/fastapi_faq.log}"

mkdir -p "$LOG_DIR"
cd "$RAG"
# shellcheck disable=SC1091
source .venv/bin/activate

export HF_HOME="${HF_HOME:-$RAG/.hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export EMBED_DEVICE="${EMBED_DEVICE:-cuda}"
export USE_RERANKER="${USE_RERANKER:-1}"
export WEATHER_AGENT_ROOT="$AGENT"
export API_HOST="${API_HOST:-0.0.0.0}"
export API_PORT="${API_PORT:-7860}"

echo "=== 1) Qdrant (vector DB) ==="
if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx qdrant-tita; then
  echo "qdrant-tita already running"
elif docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx qdrant-tita; then
  docker start qdrant-tita
else
  docker run -d --name qdrant-tita \
    -p 6333:6333 -p 6334:6334 \
    -v "${QDRANT_STORAGE:-$RAG/qdrant_storage}:/qdrant/storage" \
    qdrant/qdrant:v1.13.2
fi
for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:6333/readyz >/dev/null; then
    echo "Qdrant ready :6333"
    break
  fi
  sleep 1
  if [[ $i -eq 30 ]]; then
    echo "error: Qdrant not ready" >&2
    exit 1
  fi
done

echo "=== 2) Qwen llama-server (optional) ==="
if [[ "${START_LLM:-0}" == "1" ]]; then
  if curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; then
    echo "llama already healthy on :8080"
  else
    "$OLLM/scripts/qwen_llama_start.sh" instruct
  fi
else
  echo "skipped (set START_LLM=1 to start Qwen on :8080)"
  echo "  note: FAQ answers that need LLM require :8080; weather + skip_llm work without it"
fi

echo "=== 3) FAQ API (embed + reranker + weather tool) ==="
if [[ -f "$PIDFILE_API" ]] && kill -0 "$(cat "$PIDFILE_API")" 2>/dev/null; then
  echo "API already running pid=$(cat "$PIDFILE_API")"
else
  # free port if stale
  if ss -lntp 2>/dev/null | grep -q ":${API_PORT} "; then
    echo "error: port $API_PORT in use — run ./scripts/stop_stack.sh first" >&2
    exit 1
  fi
  nohup env \
    HF_HOME="$HF_HOME" \
    HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
    TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE" \
    EMBED_DEVICE="$EMBED_DEVICE" \
    USE_RERANKER="$USE_RERANKER" \
    WEATHER_AGENT_ROOT="$WEATHER_AGENT_ROOT" \
    API_HOST="$API_HOST" \
    API_PORT="$API_PORT" \
    python3 -u scripts/fastapi_faq_app.py \
    >>"$LOG_API" 2>&1 &
  echo $! >"$PIDFILE_API"
  echo "API pid=$(cat "$PIDFILE_API")  log=$LOG_API"
fi

for i in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:${API_PORT}/health" >/dev/null; then
    echo
    echo "Ready:"
    echo "  Qdrant   http://127.0.0.1:6333"
    echo "  FAQ API  http://127.0.0.1:${API_PORT}/ask   docs=/docs"
    [[ "${START_LLM:-0}" == "1" ]] && echo "  LLM      http://127.0.0.1:8080/v1"
    echo
    curl -s "http://127.0.0.1:${API_PORT}/health" || true
    echo
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true
    exit 0
  fi
  if ! kill -0 "$(cat "$PIDFILE_API")" 2>/dev/null; then
    echo "error: API exited — see $LOG_API" >&2
    tail -40 "$LOG_API" >&2 || true
    exit 1
  fi
  sleep 1
done

echo "error: API health timeout — see $LOG_API" >&2
tail -40 "$LOG_API" >&2 || true
exit 1
