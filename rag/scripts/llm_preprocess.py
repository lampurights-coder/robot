#!/usr/bin/env python3
"""Step 1b: Use local llama.cpp (OpenAI-compatible) to normalize Q&A chunks.

For each raw chunk, the LLM returns clean JSON:
  primary_question, question_variants[], answer, keywords[]

Also builds eval holdout candidates (paraphrase queries).

Env:
  LLM_BASE_URL=http://127.0.0.1:8080/v1
  LLM_MODEL=Qwen3.8-27B
  LIMIT=0   # 0 = all
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import httpx
from tqdm import tqdm

RAG = Path(__file__).resolve().parents[1]
DATA = RAG / "data"
RAW = DATA / "chunks_raw.jsonl"
OUT = DATA / "chunks_qa.jsonl"
EVAL_SEED = DATA / "eval_seed.jsonl"

BASE = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")
MODEL = os.environ.get("LLM_MODEL", "Qwen3.8-27B")
LIMIT = int(os.environ.get("LIMIT", "0"))

SYSTEM = """تو یک ویرایشگر داده FAQ دندانپزشکی هستی.
ورودی یک بخش FAQ با چند سؤال و یک پاسخ است.
فقط یک JSON معتبر برگردان (بدون markdown) با کلیدهای:
{
  "primary_question": "بهترین فرم رسمی سؤال",
  "question_variants": ["سایر فرم‌های سؤال"],
  "answer": "پاسخ تمیز و کامل به فارسی",
  "keywords": ["کلیدواژه1", "کلیدواژه2"],
  "paraphrase_query": "یک بازنویسی متفاوت از سؤال برای تست بازیابی"
}
قوانین: متن را اختراع نکن؛ فقط منظم و خوانا کن. غلط‌های واضح املایی را اصلاح کن (مثل پالک→پلاک، الیه→لایه، الزم→لازم).
اگر پاسخ خالی است answer را خالی بگذار."""


def extract_json(text: str) -> dict:
    text = text.strip()
    # strip think tags if any
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"no json in: {text[:200]}")
    return json.loads(m.group(0))


def call_llm(client: httpx.Client, chunk: dict) -> dict:
    user = json.dumps(
        {
            "topic": chunk["topic"],
            "questions": chunk["questions"],
            "answer": chunk["answer"],
        },
        ensure_ascii=False,
    )
    payload = {
        "model": MODEL,
        "temperature": 0.2,
        "top_p": 0.8,
        "max_tokens": 1200,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
    }
    r = client.post(f"{BASE}/chat/completions", json=payload, timeout=180.0)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    content = msg.get("content") or msg.get("reasoning_content") or ""
    return extract_json(content)


def fallback(chunk: dict) -> dict:
    qs = chunk["questions"] or ["(بدون سؤال)"]
    return {
        "primary_question": qs[0],
        "question_variants": qs[1:],
        "answer": chunk["answer"],
        "keywords": [],
        "paraphrase_query": qs[0],
        "llm_ok": False,
    }


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in RAW.read_text(encoding="utf-8").splitlines() if l.strip()]
    if LIMIT > 0:
        rows = rows[:LIMIT]

    done_ids = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["chunk_id"])

    client = httpx.Client()
    # health
    h = client.get(BASE.replace("/v1", "") + "/health", timeout=10)
    h.raise_for_status()

    mode = "a" if done_ids else "w"
    n_ok = n_fail = 0
    with OUT.open(mode, encoding="utf-8") as fout, EVAL_SEED.open(
        mode, encoding="utf-8"
    ) as feval:
        for chunk in tqdm(rows, desc="llm-preprocess"):
            if chunk["chunk_id"] in done_ids:
                continue
            try:
                llm = call_llm(client, chunk)
                llm["llm_ok"] = True
                n_ok += 1
            except Exception as e:
                llm = fallback(chunk)
                llm["error"] = str(e)[:200]
                n_fail += 1

            rec = {
                **chunk,
                "primary_question": llm.get("primary_question") or (chunk["questions"][:1] or [""])[0],
                "question_variants": llm.get("question_variants") or chunk["questions"][1:],
                "answer_clean": llm.get("answer") or chunk["answer"],
                "keywords": llm.get("keywords") or [],
                "paraphrase_query": llm.get("paraphrase_query")
                or (chunk["questions"][:1] or [""])[0],
                "llm_ok": llm.get("llm_ok", False),
            }
            # embedding text = questions + answer (retrieval-friendly)
            q_all = [rec["primary_question"], *rec["question_variants"]]
            q_all = [q for q in q_all if q]
            rec["embed_text"] = (
                "سؤالات: " + " | ".join(q_all) + "\nپاسخ: " + (rec["answer_clean"] or "")
            )
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            feval.write(
                json.dumps(
                    {
                        "chunk_id": rec["chunk_id"],
                        "query": rec["paraphrase_query"],
                        "gold_chunk_id": rec["chunk_id"],
                        "topic": rec["topic"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            feval.flush()
            time.sleep(0.05)

    print(f"done ok={n_ok} fail={n_fail} -> {OUT}")


if __name__ == "__main__":
    main()
