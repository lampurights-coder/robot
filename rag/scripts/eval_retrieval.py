#!/usr/bin/env python3
"""Step 3: Evaluate retrieval (Recall@k, MRR) on held-out paraphrase queries.

Uses the same Jina model with prompt_name=query for queries.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch
from qdrant_client import QdrantClient
from tqdm import tqdm

RAG = Path(__file__).resolve().parents[1]
DATA = RAG / "data"
SEED = DATA / "eval_seed.jsonl"
OUT = DATA / "eval_results.json"

QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "tita_faq_qa")
MODEL_ID = os.environ.get("EMBED_MODEL", "jinaai/jina-embeddings-v5-text-small")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "1024"))
EVAL_N = int(os.environ.get("EVAL_N", "80"))
KS = [1, 3, 5, 10]


def main() -> None:
    rows = [json.loads(l) for l in SEED.read_text(encoding="utf-8").splitlines() if l.strip()]
    random.seed(42)
    if len(rows) > EVAL_N:
        rows = random.sample(rows, EVAL_N)

    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {"trust_remote_code": True, "device": device}
    if device == "cuda":
        kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
    model = SentenceTransformer(MODEL_ID, **kwargs)
    client = QdrantClient(url=QDRANT_URL)

    hits_at = {k: 0 for k in KS}
    mrr_sum = 0.0
    details = []

    for row in tqdm(rows, desc="eval"):
        q = row["query"]
        gold = row["gold_chunk_id"]
        vec = model.encode(
            [q],
            task="retrieval",
            prompt_name="query",
            normalize_embeddings=True,
            truncate_dim=EMBED_DIM if EMBED_DIM < 1024 else None,
        )[0]
        res = client.search(
            collection_name=COLLECTION,
            query_vector=vec.tolist(),
            limit=max(KS),
            with_payload=True,
        )
        ranked = [p.payload["chunk_id"] for p in res]
        rank = ranked.index(gold) + 1 if gold in ranked else None
        for k in KS:
            if rank is not None and rank <= k:
                hits_at[k] += 1
        if rank is not None:
            mrr_sum += 1.0 / rank
        details.append(
            {
                "query": q,
                "gold": gold,
                "rank": rank,
                "top3": ranked[:3],
            }
        )

    n = len(rows)
    metrics = {
        "n": n,
        "recall": {f"@{k}": hits_at[k] / n for k in KS},
        "mrr": mrr_sum / n,
        "model": MODEL_ID,
        "collection": COLLECTION,
    }
    OUT.write_text(json.dumps({"metrics": metrics, "details": details}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
