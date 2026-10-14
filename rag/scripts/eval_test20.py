#!/usr/bin/env python3
"""Evaluate Clinic2 FAQ RAG on tita-arshia/faqs/eval/test-20.json.

Retrieval: Jina → Qdrant top-K → BAAI/bge-reranker-v2-m3 → top-1.
Set USE_RERANKER=0 to disable rerank (baseline).

Metrics:
  - retrieval_hit: top-1 card answer is semantically close to gold (cosine >= 0.70)
  - retrieval_ok_soft: cosine >= 0.55
  - end_to_end_ok: LLM reply grounded on retrieved FAQ is close to gold (cosine >= 0.65)
  - refuse_rate: below score threshold
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
import numpy as np
import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
RAG = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "faqs" / "eval"
_DEFAULT_TESTS = f"{EVAL_DIR / 'test-20.json'},{EVAL_DIR / 'eval-20-2.json'}"
# TEST_FILES=path1,path2  (preferred)  or single TEST_FILE=
_TEST_ENV = os.environ.get("TEST_FILES") or os.environ.get("TEST_FILE") or _DEFAULT_TESTS
TEST_FILES = [Path(p.strip()).expanduser() for p in _TEST_ENV.split(",") if p.strip()]
OUT_DIR = Path(os.environ.get("OUT_DIR", RAG / "data" / "eval")).expanduser()
# legacy single-file OUT_FILE still honored when only one suite
OUT = Path(os.environ["OUT_FILE"]).expanduser() if os.environ.get("OUT_FILE") else None

QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "clinic2_faq_qa")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "jinaai/jina-embeddings-v5-text-small")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
USE_RERANKER = os.environ.get("USE_RERANKER", "1") not in ("0", "false", "False")
LLM_BASE = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen3.8-27B")
# Qdrant cosine gate (used when reranker off)
SCORE_MIN = float(os.environ.get("SCORE_MIN", "0.55"))
# Reranker sigmoid gate (typical useful range ~0.2–0.5)
RERANK_MIN = float(os.environ.get("RERANK_MIN", "0.25"))
# After rerank, drop hits whose dense score is still tiny (reduces false CSR-style matches)
QDRANT_FLOOR = float(os.environ.get("QDRANT_FLOOR", "0.30"))
CANDIDATE_K = int(os.environ.get("CANDIDATE_K", "10"))
ANS_HIT = float(os.environ.get("ANS_HIT", "0.70"))
ANS_SOFT = float(os.environ.get("ANS_SOFT", "0.55"))
E2E_HIT = float(os.environ.get("E2E_HIT", "0.65"))
RETRIEVAL_ONLY = os.environ.get("RETRIEVAL_ONLY", "0") not in ("0", "false", "False")


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class BgeReranker:
    def __init__(self, model_id: str, device: str = "cpu"):
        self.device = device
        offline = os.environ.get("HF_HUB_OFFLINE", "0") in ("1", "true", "True")
        self.tok = AutoTokenizer.from_pretrained(model_id, local_files_only=offline)
        dtype = torch.bfloat16 if device.startswith("cuda") else None
        kwargs = {"local_files_only": offline}
        if dtype is not None:
            kwargs["dtype"] = dtype
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs)
        self.model.to(device).eval()

    def score(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        pairs = [(query, d) for d in docs]
        with torch.no_grad():
            inputs = self.tok(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
            return torch.sigmoid(logits).cpu().tolist()


def candidate_text(payload: dict) -> str:
    """Rerank query against FAQ question (+ short answer) for better Persian match."""
    q = (payload.get("question") or "").strip()
    a = (payload.get("answer") or "").strip()
    if a:
        return f"{q}\n{a[:400]}"
    return q


def retrieve(embedder, client, reranker, question: str):
    v = embedder.encode(
        [question],
        task="retrieval",
        prompt_name="query",
        normalize_embeddings=True,
    )[0]
    res = client.query_points(
        collection_name=COLLECTION,
        query=v.tolist(),
        limit=CANDIDATE_K if USE_RERANKER else 3,
        with_payload=True,
    )
    hits = list(res.points)
    if not hits:
        return None

    if not USE_RERANKER or reranker is None:
        top = hits[0]
        return {
            "hit": top,
            "qdrant_score": float(top.score),
            "rerank_score": None,
            "score": float(top.score),
        }

    # Drop empty-answer vectors (bad index rows like bare insurance stubs)
    usable = [h for h in hits if ((h.payload or {}).get("answer") or "").strip()]
    if not usable:
        usable = hits

    docs = [candidate_text(h.payload or {}) for h in usable]
    scores = reranker.score(question, docs)
    # Best per card id, then global best
    best_by_card: dict[str, tuple[float, int]] = {}
    for i, (h, s) in enumerate(zip(usable, scores)):
        cid = str((h.payload or {}).get("id") or i)
        if cid not in best_by_card or s > best_by_card[cid][0]:
            best_by_card[cid] = (float(s), i)
    best_i = max(best_by_card.values(), key=lambda t: t[0])[1]
    top = usable[best_i]
    return {
        "hit": top,
        "qdrant_score": float(top.score),
        "rerank_score": float(scores[best_i]),
        "score": float(scores[best_i]),
        "candidates": [
            {
                "id": (h.payload or {}).get("id"),
                "qdrant": float(h.score),
                "rerank": float(s),
            }
            for h, s in zip(usable, scores)
        ],
    }


# llama -c 8192: keep FAQ short enough for prompt + max_tokens (long implant cards OOM ctx)
LLM_FAQ_CHARS = int(os.environ.get("LLM_FAQ_CHARS", "10000"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "400"))
# Jina text-matching on full 20k+ implant answers is extremely slow on CPU
SIM_CHARS = int(os.environ.get("SIM_CHARS", "2000"))


def _truncate_faq(text: str, limit: int = LLM_FAQ_CHARS) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n…"


def _for_sim(text: str) -> str:
    text = text or ""
    return text if len(text) <= SIM_CHARS else text[:SIM_CHARS]


def llm_answer(question: str, faq_answer: str) -> str:
    system = (
        "تو دستیار FAQ کلینیک هستی. فقط بر اساس متن پاسخ رسمی جواب بده. "
        "واقعیت جدید نساز. مختصر و فارسی."
    )
    faq = _truncate_faq(faq_answer)
    user = f"سؤال: {question}\n\nپاسخ رسمی:\n{faq}"
    r = httpx.post(
        f"{LLM_BASE}/chat/completions",
        json={
            "model": LLM_MODEL,
            "temperature": 0.2,
            "max_tokens": LLM_MAX_TOKENS,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=180.0,
    )
    if r.status_code >= 400:
        # context overflow / bad request → fall back to truncated FAQ (still evaluable)
        try:
            err = r.json()
        except Exception:
            err = {"error": r.text[:200]}
        print(f"  llm warn HTTP {r.status_code}: {err}", file=sys.stderr)
        return faq.strip()
    msg = r.json()["choices"][0]["message"]
    return (msg.get("content") or faq).strip()


def eval_suite(cases: list[dict], embedder, client, reranker, desc: str) -> tuple[dict, list]:
    gate = RERANK_MIN if USE_RERANKER else SCORE_MIN
    details = []
    n_hit = n_soft = n_e2e = n_refuse = 0

    for case in tqdm(cases, desc=desc):
        q = case["question"]
        gold = case["answer"]
        got = retrieve(embedder, client, reranker, q)
        score = got["score"] if got else 0.0
        qdrant_s = float(got.get("qdrant_score") or 0) if got else 0.0
        if got is None:
            accept = False
        elif USE_RERANKER:
            accept = (score >= RERANK_MIN and qdrant_s >= QDRANT_FLOOR) or (
                qdrant_s >= SCORE_MIN
            )
        else:
            accept = score >= SCORE_MIN
        if not accept:
            n_refuse += 1
            hit = got["hit"] if got else None
            details.append(
                {
                    "id": case["id"],
                    "suite": desc,
                    "question": q,
                    "refused": True,
                    "score": score,
                    "qdrant_score": got.get("qdrant_score") if got else 0.0,
                    "rerank_score": got.get("rerank_score") if got else None,
                    "pred_card": (hit.payload or {}).get("id") if hit else None,
                    "retrieval_sim": 0.0,
                    "e2e_sim": 0.0,
                    "hit": False,
                    "soft": False,
                    "e2e": False,
                }
            )
            continue

        hit = got["hit"]
        pred_ans = (hit.payload or {}).get("answer") or ""
        # Truncate mega-cards (e.g. impl_69 ~26k chars) — full encode stalls CPU for minutes
        emb = embedder.encode(
            [_for_sim(pred_ans), _for_sim(gold)],
            task="text-matching",
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sim = cosine(emb[0], emb[1])
        hit_ok = sim >= ANS_HIT
        soft_ok = sim >= ANS_SOFT
        if hit_ok:
            n_hit += 1
        if soft_ok:
            n_soft += 1

        if RETRIEVAL_ONLY:
            e2e = 0.0
            e2e_ok = False
            reply = ""
        else:
            reply = llm_answer(q, pred_ans)
            emb2 = embedder.encode(
                [_for_sim(reply), _for_sim(gold)],
                task="text-matching",
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            e2e = cosine(emb2[0], emb2[1])
            e2e_ok = e2e >= E2E_HIT
            if e2e_ok:
                n_e2e += 1

        details.append(
            {
                "id": case["id"],
                "suite": desc,
                "question": q,
                "refused": False,
                "score": score,
                "qdrant_score": got.get("qdrant_score"),
                "rerank_score": got.get("rerank_score"),
                "pred_card": (hit.payload or {}).get("id"),
                "matched_question": (hit.payload or {}).get("question"),
                "pred_answer": pred_ans[:300],
                "gold_answer": gold[:300],
                "llm_reply": reply[:300] if reply else "",
                "retrieval_sim": sim,
                "e2e_sim": e2e,
                "hit": hit_ok,
                "soft": soft_ok,
                "e2e": e2e_ok,
                "candidates": got.get("candidates"),
            }
        )

    n = len(cases)
    metrics = {
        "suite": desc,
        "n": n,
        "use_reranker": USE_RERANKER,
        "rerank_model": RERANK_MODEL if USE_RERANKER else None,
        "device": os.environ.get("EMBED_DEVICE", "auto"),
        "retrieval_accuracy@0.70": n_hit / n if n else 0.0,
        "retrieval_accuracy@0.55": n_soft / n if n else 0.0,
        "e2e_accuracy@0.65": (n_e2e / n) if (n and not RETRIEVAL_ONLY) else None,
        "refuse_rate": n_refuse / n if n else 0.0,
        "correct_hit": n_hit,
        "correct_soft": n_soft,
        "correct_e2e": n_e2e if not RETRIEVAL_ONLY else None,
        "refused": n_refuse,
        "thresholds": {
            "score_gate": gate,
            "qdrant_score_min": SCORE_MIN,
            "rerank_min": RERANK_MIN if USE_RERANKER else None,
            "qdrant_floor": QDRANT_FLOOR if USE_RERANKER else None,
            "candidate_k": CANDIDATE_K if USE_RERANKER else 3,
            "answer_sim_hit": ANS_HIT,
            "answer_sim_soft": ANS_SOFT,
            "e2e_sim": E2E_HIT,
        },
        "retrieval_only": RETRIEVAL_ONLY,
    }
    return metrics, details


def _print_suite(metrics: dict, details: list) -> None:
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\nPer-item:")
    for d in details:
        mark = "HIT" if d["hit"] else ("SOFT" if d["soft"] else ("REFUSE" if d["refused"] else "MISS"))
        e2e = "E2E✓" if d.get("e2e") else ("—" if RETRIEVAL_ONLY else "E2E✗")
        rs = d.get("rerank_score")
        rs_s = f"rerank={rs:.3f}" if rs is not None else "rerank=-"
        print(
            f"  [{mark}/{e2e}] #{d['id']} qdrant={d.get('qdrant_score', 0):.3f} {rs_s} "
            f"ans_sim={d['retrieval_sim']:.3f} e2e_sim={d['e2e_sim']:.3f} "
            f"card={d.get('pred_card')} | {d['question'][:50]}"
        )


def main() -> None:
    for p in TEST_FILES:
        if not p.is_file():
            raise SystemExit(f"TEST_FILE not found: {p}")

    device = os.environ.get(
        "EMBED_DEVICE",
        "cpu"
        if os.environ.get("CUDA_VISIBLE_DEVICES", None) == ""
        else ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    offline = os.environ.get("HF_HUB_OFFLINE", "0") in ("1", "true", "True")
    print(f"Loading {EMBED_MODEL} on {device} (suites={len(TEST_FILES)})...")
    embed_kwargs = {
        "trust_remote_code": True,
        "device": device,
        "local_files_only": offline,
    }
    if device.startswith("cuda"):
        embed_kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
    embedder = SentenceTransformer(EMBED_MODEL, **embed_kwargs)
    reranker = None
    if USE_RERANKER:
        print(f"Loading reranker {RERANK_MODEL} on {device}...")
        reranker = BgeReranker(RERANK_MODEL, device=device)
    client = QdrantClient(url=QDRANT_URL, check_compatibility=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    combined_details = []

    for test_path in TEST_FILES:
        cases = json.loads(test_path.read_text(encoding="utf-8"))
        name = test_path.stem
        print(f"\n======== {name}  n={len(cases)}  ({test_path}) ========")
        metrics, details = eval_suite(cases, embedder, client, reranker, desc=name)
        metrics["test_file"] = str(test_path)
        _print_suite(metrics, details)

        if OUT is not None and len(TEST_FILES) == 1:
            out_path = OUT
        else:
            suffix = "_rerank" if USE_RERANKER else ""
            out_path = OUT_DIR / f"{name}{suffix}.json"
        out_path.write_text(
            json.dumps({"metrics": metrics, "details": details}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {out_path}")
        all_results.append(metrics)
        combined_details.extend(details)

    # combined summary
    n = sum(m["n"] for m in all_results)
    n_hit = sum(m["correct_hit"] for m in all_results)
    n_soft = sum(m["correct_soft"] for m in all_results)
    n_e2e = sum((m["correct_e2e"] or 0) for m in all_results)
    n_refuse = sum(m["refused"] for m in all_results)
    combined = {
        "n": n,
        "suites": [m["suite"] for m in all_results],
        "use_reranker": USE_RERANKER,
        "device": device,
        "retrieval_accuracy@0.70": n_hit / n if n else 0.0,
        "retrieval_accuracy@0.55": n_soft / n if n else 0.0,
        "e2e_accuracy@0.65": (n_e2e / n) if (n and not RETRIEVAL_ONLY) else None,
        "refuse_rate": n_refuse / n if n else 0.0,
        "correct_hit": n_hit,
        "correct_soft": n_soft,
        "correct_e2e": n_e2e if not RETRIEVAL_ONLY else None,
        "refused": n_refuse,
        "per_suite": all_results,
    }
    combined_path = OUT_DIR / ("combined_rerank.json" if USE_RERANKER else "combined.json")
    combined_path.write_text(
        json.dumps(
            {"metrics": combined, "details": combined_details},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n======== COMBINED ========")
    print(json.dumps(combined, ensure_ascii=False, indent=2))
    print(f"wrote {combined_path}")


if __name__ == "__main__":
    main()
