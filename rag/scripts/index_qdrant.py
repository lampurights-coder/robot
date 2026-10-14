#!/usr/bin/env python3
"""Step 2: Embed Q&A chunks with jina-embeddings-v5-text-small and upsert to Qdrant.

Collection: tita_faq_qa
Vector: 1024-dim (Matryoshka; truncate_dim optional via EMBED_DIM)
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from tqdm import tqdm

RAG = Path(__file__).resolve().parents[1]
DATA = RAG / "data"
QA = DATA / "chunks_qa.jsonl"

QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "tita_faq_qa")
MODEL_ID = os.environ.get("EMBED_MODEL", "jinaai/jina-embeddings-v5-text-small")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "1024"))
BATCH = int(os.environ.get("EMBED_BATCH", "8"))


def load_model():
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {"trust_remote_code": True, "device": device}
    if device == "cuda":
        kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
    model = SentenceTransformer(MODEL_ID, **kwargs)
    return model


def embed(model, texts: list[str]):
    # retrieval document side
    return model.encode(
        texts,
        task="retrieval",
        prompt_name="document",
        normalize_embeddings=True,
        truncate_dim=EMBED_DIM if EMBED_DIM < 1024 else None,
        show_progress_bar=False,
        batch_size=BATCH,
    )


def main() -> None:
    rows = [json.loads(l) for l in QA.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"loading {len(rows)} chunks; model={MODEL_ID} dim={EMBED_DIM}")
    model = load_model()
    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=qm.VectorParams(size=EMBED_DIM, distance=qm.Distance.COSINE),
    )

    points = []
    for i in tqdm(range(0, len(rows), BATCH), desc="embed+upsert"):
        batch = rows[i : i + BATCH]
        vecs = embed(model, [r["embed_text"] for r in batch])
        for r, v in zip(batch, vecs):
            pid = str(uuid.uuid5(uuid.NAMESPACE_URL, r["chunk_id"]))
            points.append(
                qm.PointStruct(
                    id=pid,
                    vector=v.tolist(),
                    payload={
                        "chunk_id": r["chunk_id"],
                        "topic": r["topic"],
                        "source_file": r["source_file"],
                        "section_id": r["section_id"],
                        "start_line": r["start_line"],
                        "primary_question": r["primary_question"],
                        "question_variants": r.get("question_variants") or [],
                        "answer": r.get("answer_clean") or r.get("answer") or "",
                        "keywords": r.get("keywords") or [],
                        "embed_text": r["embed_text"],
                    },
                )
            )
        client.upsert(collection_name=COLLECTION, points=points[-len(batch) :])

    info = client.get_collection(COLLECTION)
    print(f"indexed collection={COLLECTION} points={info.points_count}")


if __name__ == "__main__":
    main()
