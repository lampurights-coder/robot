#!/usr/bin/env bash
# Full RAG pipeline for tita-arshia FAQs
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 scripts/parse_faq_chunks.py

# Optional: LIMIT=20 for a smoke test
LIMIT="${LIMIT:-0}" LIMIT="$LIMIT" python3 scripts/llm_preprocess.py

python3 scripts/index_qdrant.py
EVAL_N="${EVAL_N:-80}" python3 scripts/eval_retrieval.py

echo "Artifacts in $ROOT/data/"
ls -la "$ROOT/data/"
