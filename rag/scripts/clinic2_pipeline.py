#!/usr/bin/env python3
"""FAQ-Clinic2 only: clean → cards → Q/A → embed each QUESTION → Qdrant → ask + eval.

Follows the card model:
  - Search by questions (each question = one vector)
  - Payload carries the same answer for all questions on that card
  - Do NOT use LLM-rewritten answers (medical safety)
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

ROOT = Path(__file__).resolve().parents[2]
RAG = Path(__file__).resolve().parents[1]
SRC = ROOT / "faqs" / "sources" / "clinic2.txt"
OUT = RAG / "data" / "clinic2"
COLLECTION = "clinic2_faq_qa"
QDRANT_URL = "http://127.0.0.1:6333"
MODEL_ID = "jinaai/jina-embeddings-v5-text-small"
EMBED_DIM = 1024
SCORE_MIN = 0.55

# PDF letter glitches (لا → ال etc.) — extend as found
FIXES = {
    "اصطالحات": "اصطلاحات",
    "اصطالحا": "اصطلاحا",
    "تالش": "تلاش",
    "قبالً": "قبلاً",
    "قبال": "قبلا",
    "الزم": "لازم",
    "الیهای": "لایه‌ای",
    "الیه": "لایه",
    "پالک": "پلاک",
    "مالحظات": "ملاحظات",
    "مالحظاتی": "ملاحظاتی",
    "اطالع": "اطلاع",
    "طوالنی": "طولانی",
    "معموالً": "معمولاً",
    "معموال": "معمولا",
    "باال": "بالا",
    "مثالً": "مثلاً",
    "عالئم": "علائم",
    "ریفالکس": "ریفلاکس",
}


def clean_text(text: str) -> str:
    text = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]", "", text)
    for bad, good in FIXES.items():
        text = text.replace(bad, good)
    return text


def split_cards(text: str) -> list[tuple[str, str]]:
    """Split on '- N ' at line start (Clinic2 style)."""
    # Normalize "؟" stuck to next sentence
    parts = re.split(r"(?m)^\s*-\s*(\d+)\s+", text)
    # parts[0]=preamble, then id, body, id, body...
    cards = []
    for i in range(1, len(parts) - 1, 2):
        cards.append((parts[i], parts[i + 1]))
    return cards


def parse_card(body: str) -> tuple[list[str], str]:
    """Questions end with ؟; answer = after last question."""
    body = " ".join(body.split())
    questions: list[str] = []
    while True:
        m = re.match(r"^(.{3,200}?[؟?])\s*", body)
        if not m:
            break
        q = m.group(1).strip()
        # skip tiny garbage
        if len(q) >= 5:
            questions.append(q)
        body = body[m.end() :]
    answer = body.strip()
    return questions, answer


def build_records() -> list[dict]:
    raw = SRC.read_text(encoding="utf-8")
    text = clean_text(raw)
    # drop title line
    text = re.sub(r"(?m)^FAQ-Clinic2?\s*$", "", text)
    cards = split_cards(text)
    records = []
    for num, body in cards:
        questions, answer = parse_card(body)
        rec = {
            "id": f"clinic2_{num}",
            "section": int(num),
            "questions": questions,
            "answer": answer,
            "n_questions": len(questions),
            "answer_len": len(answer),
        }
        records.append(rec)
    return records


def print_review(records: list[dict]) -> None:
    print(f"Parsed {len(records)} cards from FAQ-Clinic2")
    nums = [r["section"] for r in records]
    print(f"Section numbers: {nums}")
    missing = [i for i in range(1, max(nums) + 1) if i not in nums]
    if missing:
        print(f"WARNING missing sections: {missing}")
    bad = [r for r in records if r["n_questions"] == 0 or r["answer_len"] < 20]
    print(f"Needs review (0 Q or short A): {len(bad)}")
    for r in bad:
        print(f"  - {r['id']}: qs={r['n_questions']} ans_len={r['answer_len']}")
    # show card 1 (the cost one user cares about)
    c1 = next((r for r in records if r["section"] == 1), None)
    if c1:
        print("\n=== card clinic2_1 (costs) ===")
        for q in c1["questions"]:
            print(f"  Q: {q}")
        print(f"  A: {c1['answer'][:300]}...")


def index_questions(records: list[dict]) -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} on {device}...")
    kwargs = {"trust_remote_code": True, "device": device}
    if device == "cuda":
        kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
    model = SentenceTransformer(MODEL_ID, **kwargs)

    client = QdrantClient(url=QDRANT_URL, check_compatibility=False)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config=qm.VectorParams(size=EMBED_DIM, distance=qm.Distance.COSINE),
    )

    points = []
    for rec in records:
        if not rec["questions"]:
            continue
        # document/passage side for FAQ questions stored as passages
        vecs = model.encode(
            rec["questions"],
            task="retrieval",
            prompt_name="document",
            normalize_embeddings=True,
            batch_size=16,
            show_progress_bar=False,
        )
        for q, v in zip(rec["questions"], vecs):
            points.append(
                qm.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=v.tolist(),
                    payload={
                        "id": rec["id"],
                        "section": rec["section"],
                        "question": q,
                        "answer": rec["answer"],
                        "all_questions": rec["questions"],
                    },
                )
            )
    client.upload_points(COLLECTION, points, batch_size=64, wait=True)
    print(f"Indexed {len(points)} question vectors → collection `{COLLECTION}`")


def ask(model, client, user_question: str, limit: int = 1) -> dict:
    v = model.encode(
        [user_question],
        task="retrieval",
        prompt_name="query",
        normalize_embeddings=True,
    )[0]
    res = client.query_points(
        collection_name=COLLECTION,
        query=v.tolist(),
        limit=limit,
        with_payload=True,
    )
    hits = res.points
    if not hits:
        return {"ok": False, "reason": "no_hits"}
    top = hits[0]
    if top.score < SCORE_MIN:
        return {
            "ok": False,
            "reason": "low_score",
            "score": top.score,
            "matched_question": top.payload.get("question"),
            "card_id": top.payload.get("id"),
        }
    return {
        "ok": True,
        "score": top.score,
        "card_id": top.payload["id"],
        "matched_question": top.payload["question"],
        "answer": top.payload["answer"],
    }


def run_eval(model, client, records: list[dict]) -> dict:
    """Patient-style queries for Clinic2 — gold = card id."""
    tests = [
        # card 1 — costs (user's examples)
        ("هزینه روکش چقدره؟", "clinic2_1"),
        ("هزینه ایمپلنت چنده؟", "clinic2_1"),
        ("جرمگیری چند تومنه؟", "clinic2_1"),
        ("پر کردن دندون چقدر هزینه داره؟", "clinic2_1"),
        ("هزینه کامپوزیت هر دندون؟", "clinic2_1"),
        ("عصب کشی چقدر پول میخواد؟", "clinic2_1"),
        ("ارتودنسی چند؟", "clinic2_1"),
        # other cards if present
        ("جانبازا خدمات دندونپزشکی رایگان دارن؟", "clinic2_2"),
        ("روکشم افتاد باید دوباره وقت بگیرم؟", "clinic2_3"),
    ]
    # only keep tests whose gold card exists
    ids = {r["id"] for r in records}
    tests = [(q, g) for q, g in tests if g in ids]

    correct = 0
    details = []
    for q, gold in tests:
        r = ask(model, client, q)
        ok = r.get("ok") and r.get("card_id") == gold
        if ok:
            correct += 1
        details.append(
            {
                "query": q,
                "gold": gold,
                "pred": r.get("card_id"),
                "score": r.get("score"),
                "correct": ok,
                "matched_question": r.get("matched_question"),
            }
        )
    metrics = {
        "n": len(tests),
        "accuracy": correct / len(tests) if tests else 0.0,
        "correct": correct,
        "score_min": SCORE_MIN,
    }
    return {"metrics": metrics, "details": details}


def answer_with_llm(answer: str, user_q: str) -> str:
    """Optional: pass retrieved answer to local LLM to phrase a reply (does not invent facts)."""
    import httpx

    base = "http://127.0.0.1:8080/v1"
    try:
        httpx.get("http://127.0.0.1:8080/health", timeout=3).raise_for_status()
    except Exception:
        return answer  # LLM down → return raw answer

    system = (
        "تو دستیار کلینیک دندانپزشکی هستی. فقط بر اساس «متن پاسخ رسمی» جواب بده. "
        "هیچ واقعیت پزشکی جدیدی اضافه نکن. مختصر و مؤدب به فارسی بنویس."
    )
    user = f"سؤال بیمار: {user_q}\n\nمتن پاسخ رسمی:\n{answer}"
    r = httpx.post(
        f"{base}/chat/completions",
        json={
            "model": "Qwen3.8-27B",
            "temperature": 0.3,
            "max_tokens": 400,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=120,
    )
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    return (msg.get("content") or answer).strip()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = build_records()
    print_review(records)

    (OUT / "records.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    (OUT / "records_preview.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved {OUT / 'records.jsonl'}")

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {"trust_remote_code": True, "device": device}
    if device == "cuda":
        kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
    model = SentenceTransformer(MODEL_ID, **kwargs)
    client = QdrantClient(url=QDRANT_URL, check_compatibility=False)

    # Re-index only if collection missing / empty
    try:
        n = client.get_collection(COLLECTION).points_count or 0
    except Exception:
        n = 0
    if n == 0:
        index_questions(records)
    else:
        print(f"Reusing existing collection `{COLLECTION}` ({n} points)")

    # Demo asks for the cost questions
    demos = [
        "هزینهی روکش چقدره؟",
        "هزینه ایمپلنت چقدره؟",
        "هزینه جرمگیری چنده؟",
        "هزینه عصب کشی چقدره؟",
    ]
    print("\n=== ask demos ===")
    demo_out = []
    for q in demos:
        hit = ask(model, client, q)
        phrased = (
            answer_with_llm(hit["answer"], q) if hit.get("ok") else "پاسخی برای این سؤال ندارم."
        )
        print(f"\nQ: {q}")
        print(f"  card={hit.get('card_id')} score={hit.get('score')}")
        print(f"  matched: {hit.get('matched_question')}")
        print(f"  reply: {phrased[:240]}")
        demo_out.append({"query": q, "retrieval": hit, "llm_reply": phrased})

    (OUT / "demo_asks.json").write_text(
        json.dumps(demo_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    ev = run_eval(model, client, records)
    (OUT / "eval_results.json").write_text(
        json.dumps(ev, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== eval ===")
    print(json.dumps(ev["metrics"], ensure_ascii=False, indent=2))
    for d in ev["details"]:
        mark = "OK" if d["correct"] else "MISS"
        print(f"  [{mark}] {d['query'][:40]} → {d['pred']} (gold {d['gold']}) score={d['score']}")


if __name__ == "__main__":
    main()
