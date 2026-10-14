#!/usr/bin/env python3
"""Query the Qdrant FAQ index (Jina query embedding)."""

from __future__ import annotations

import argparse
import json
import os

import torch
from qdrant_client import QdrantClient

QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "tita_faq_qa")
MODEL_ID = os.environ.get("EMBED_MODEL", "jinaai/jina-embeddings-v5-text-small")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "1024"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {"trust_remote_code": True, "device": device}
    if device == "cuda":
        kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
    model = SentenceTransformer(MODEL_ID, **kwargs)
    vec = model.encode(
        [args.query],
        task="retrieval",
        prompt_name="query",
        normalize_embeddings=True,
        truncate_dim=EMBED_DIM if EMBED_DIM < 1024 else None,
    )[0]
    client = QdrantClient(url=QDRANT_URL)
    hits = client.search(COLLECTION, query_vector=vec.tolist(), limit=args.k, with_payload=True)
    for i, h in enumerate(hits, 1):
        p = h.payload
        print(f"#{i} score={h.score:.4f}  {p['chunk_id']}")
        print(f"  Q: {p.get('primary_question')}")
        ans = (p.get("answer") or "")[:220]
        print(f"  A: {ans}...")
        print()


if __name__ == "__main__":
    main()
