#!/usr/bin/env bash
# Ingest one FAQ source into Qdrant (GPU embeddings by default).
#
# Usage:
#   ./scripts/ingest_faq.sh restorative
#   ./scripts/ingest_faq.sh 1745874244336_FAQ-Restorative
#   ./scripts/ingest_faq.sh prosthesis
#   EMBED_DEVICE=cpu ./scripts/ingest_faq.sh implant
#
# Needs: Qdrant on :6333  (docker start qdrant-tita)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAG="$ROOT/rag"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <source|legacy-name|path.txt>" >&2
  echo "examples:" >&2
  echo "  $0 restorative" >&2
  echo "  $0 1745874244336_FAQ-Restorative" >&2
  echo "  $0 anesthesia" >&2
  echo "  $0 \"FAQ-General Anesthesia\"" >&2
  exit 1
fi

# Join all args so unquoted "FAQ-General Anesthesia" still works
SPEC="$*"

cd "$RAG"
# shellcheck disable=SC1091
source .venv/bin/activate

export HF_HOME="${HF_HOME:-$RAG/.hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export EMBED_DEVICE="${EMBED_DEVICE:-cuda}"

if ! curl -sf http://127.0.0.1:6333/readyz >/dev/null; then
  echo "error: Qdrant not ready on :6333 — run: docker start qdrant-tita" >&2
  exit 1
fi

echo "Ingesting: $SPEC  (EMBED_DEVICE=$EMBED_DEVICE)"
python3 -u scripts/faq_ingest.py "$SPEC" --device "$EMBED_DEVICE"
