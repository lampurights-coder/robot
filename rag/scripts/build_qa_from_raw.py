#!/usr/bin/env python3
"""Build chunks_qa.jsonl + eval_seed from raw parse (no LLM).

Use when you want to index immediately; later re-run llm_preprocess.py
and index_qdrant.py for cleaned variants/paraphrases.
"""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
RAW = DATA / "chunks_raw.jsonl"
OUT = DATA / "chunks_qa.jsonl"
EVAL = DATA / "eval_seed.jsonl"


def main() -> None:
    rows = [json.loads(l) for l in RAW.read_text(encoding="utf-8").splitlines() if l.strip()]
    with OUT.open("w", encoding="utf-8") as fo, EVAL.open("w", encoding="utf-8") as fe:
        for r in rows:
            qs = r.get("questions") or []
            primary = qs[0] if qs else f"{r['topic']} section {r['section_id']}"
            variants = qs[1:]
            answer = r.get("answer") or ""
            embed = "سؤالات: " + " | ".join(qs or [primary]) + "\nپاسخ: " + answer
            rec = {
                **r,
                "primary_question": primary,
                "question_variants": variants,
                "answer_clean": answer,
                "keywords": [],
                "paraphrase_query": primary,
                "llm_ok": False,
                "embed_text": embed,
            }
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if qs:
                fe.write(
                    json.dumps(
                        {
                            "chunk_id": r["chunk_id"],
                            "query": primary,
                            "gold_chunk_id": r["chunk_id"],
                            "topic": r["topic"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    print(f"wrote {len(rows)} -> {OUT} (llm_ok=False; refine with llm_preprocess.py)")


if __name__ == "__main__":
    main()
