#!/usr/bin/env bash
# Run every JSON suite under faqs/eval/ in one pass (GPU embed+rerank + LLM).
# Writes per-suite + combined JSON and a markdown report.
#
# Usage:
#   ./scripts/run_all_evals.sh
#   EMBED_DEVICE=cpu ./scripts/run_all_evals.sh
#   USE_RERANKER=0 ./scripts/run_all_evals.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAG="$ROOT/rag"
EVAL_DIR="${EVAL_DIR:-$ROOT/faqs/eval}"
OUT_DIR="${OUT_DIR:-$RAG/data/eval}"
REPORT="${REPORT:-$OUT_DIR/REPORT.md}"

cd "$RAG"
# shellcheck disable=SC1091
source .venv/bin/activate

export HF_HOME="${HF_HOME:-$RAG/.hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export EMBED_DEVICE="${EMBED_DEVICE:-cuda}"
export USE_RERANKER="${USE_RERANKER:-1}"
export OUT_DIR

mkdir -p "$OUT_DIR"

# Discover suites (stable order: test-* then eval-*)
mapfile -t SUITES < <(find "$EVAL_DIR" -maxdepth 1 -type f -name '*.json' | sort)
if [[ ${#SUITES[@]} -eq 0 ]]; then
  echo "error: no *.json in $EVAL_DIR" >&2
  exit 1
fi

# Preflight
if ! curl -sf http://127.0.0.1:6333/readyz >/dev/null; then
  echo "error: Qdrant not ready on :6333 (docker start qdrant-tita)" >&2
  exit 1
fi
if [[ "${RETRIEVAL_ONLY:-0}" != "1" ]] && ! curl -sf http://127.0.0.1:8080/health >/dev/null; then
  echo "error: llama-server not ready on :8080" >&2
  exit 1
fi

JOINED="$(IFS=,; echo "${SUITES[*]}")"
echo "Suites (${#SUITES[@]}):"
printf '  - %s\n' "${SUITES[@]}"
echo "EMBED_DEVICE=$EMBED_DEVICE  USE_RERANKER=$USE_RERANKER  OUT_DIR=$OUT_DIR"
echo

TEST_FILES="$JOINED" python3 -u scripts/eval_test20.py | tee "$OUT_DIR/all_evals_run.log"

# Markdown report from combined JSON
COMBINED="$OUT_DIR/combined_rerank.json"
if [[ "${USE_RERANKER}" == "0" || "${USE_RERANKER}" == "false" ]]; then
  COMBINED="$OUT_DIR/combined.json"
fi

python3 - <<'PY' "$COMBINED" "$REPORT"
import json, sys
from datetime import datetime, timezone
from pathlib import Path

combined_path, report_path = Path(sys.argv[1]), Path(sys.argv[2])
if not combined_path.is_file():
    raise SystemExit(f"missing combined results: {combined_path}")

data = json.loads(combined_path.read_text(encoding="utf-8"))
m = data["metrics"]
details = data.get("details") or []

lines = []
lines.append("# FAQ RAG eval report")
lines.append("")
lines.append(f"- Generated: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`")
lines.append(f"- Device: `{m.get('device')}`")
lines.append(f"- Reranker: `{m.get('use_reranker')}`")
lines.append(f"- Suites: {', '.join(f'`{s}`' for s in m.get('suites') or [])}")
lines.append("")
lines.append("## Combined")
lines.append("")
lines.append("| Metric | Score |")
lines.append("|--------|------:|")
n = m["n"]
lines.append(f"| n | {n} |")
lines.append(f"| Retrieval @0.70 | {m['correct_hit']}/{n} ({100*m['retrieval_accuracy@0.70']:.1f}%) |")
lines.append(f"| Retrieval @0.55 | {m['correct_soft']}/{n} ({100*m['retrieval_accuracy@0.55']:.1f}%) |")
e2e = m.get("e2e_accuracy@0.65")
if e2e is not None:
    lines.append(f"| E2E @0.65 | {m['correct_e2e']}/{n} ({100*e2e:.1f}%) |")
lines.append(f"| Refused | {m['refused']}/{n} ({100*m['refuse_rate']:.1f}%) |")
lines.append("")
lines.append("## Per suite")
lines.append("")
lines.append("| Suite | n | Ret@0.70 | Ret@0.55 | E2E@0.65 | Refused |")
lines.append("|-------|--:|--------:|--------:|---------:|--------:|")
for s in m.get("per_suite") or []:
    sn = s["n"]
    e2e_s = s.get("e2e_accuracy@0.65")
    e2e_cell = f"{s.get('correct_e2e')}/{sn} ({100*e2e_s:.0f}%)" if e2e_s is not None else "—"
    lines.append(
        f"| `{s['suite']}` | {sn} | "
        f"{s['correct_hit']}/{sn} ({100*s['retrieval_accuracy@0.70']:.0f}%) | "
        f"{s['correct_soft']}/{sn} ({100*s['retrieval_accuracy@0.55']:.0f}%) | "
        f"{e2e_cell} | "
        f"{s['refused']}/{sn} |"
    )
lines.append("")

# misses / refuses
bad = [d for d in details if d.get("refused") or not d.get("e2e")]
lines.append("## Misses / refuses")
lines.append("")
if not bad:
    lines.append("_None._")
else:
    lines.append("| Suite | # | Status | Card | Question |")
    lines.append("|-------|--:|--------|------|----------|")
    for d in bad:
        st = "REFUSE" if d.get("refused") else ("E2E✗" if not d.get("e2e") else "MISS")
        if d.get("hit") is False and not d.get("refused") and d.get("soft"):
            st = "SOFT/E2E✗" if not d.get("e2e") else "SOFT"
        q = (d.get("question") or "").replace("|", "/")[:60]
        lines.append(
            f"| `{d.get('suite')}` | {d.get('id')} | {st} | `{d.get('pred_card')}` | {q} |"
        )
lines.append("")
lines.append("## Artifacts")
lines.append("")
lines.append(f"- Combined JSON: `{combined_path}`")
lines.append(f"- Per-suite JSON: `{combined_path.parent}/*_rerank.json`")
lines.append("")

report_path.write_text("\n".join(lines), encoding="utf-8")
print(f"\nReport → {report_path}")
print("\n".join(lines[:40]))
PY

echo
echo "Done. Report: $REPORT"
