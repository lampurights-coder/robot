# Tita-Arshia FAQ RAG (Jina v5-small + Qdrant)

Pipeline: FAQ sources → Q/A cards → embeddings → Qdrant → (optional) BGE rerank → local Qwen.

Embedding: [jinaai/jina-embeddings-v5-text-small](https://huggingface.co/jinaai/jina-embeddings-v5-text-small)  
Reranker: [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)

## Layout

```
tita-arshia/
  faqs/sources/          # endo.txt, clinic2.txt, …
  faqs/eval/test-20.json
  rag/
    scripts/
      clinic2_pipeline.py    # Clinic2 → Qdrant
      endo_pipeline.py       # Endo → append Qdrant (GPU)
      eval_test20.py         # accuracy eval (+ rerank)
      gradio_faq_app.py      # UI (LangGraph)
    data/clinic2/  data/endo/
```

## Run

```bash
docker start qdrant-tita || docker run -d --name qdrant-tita -p 6333:6333 \
  -v "$PWD/qdrant_storage:/qdrant/storage" qdrant/qdrant:v1.13.2

cd /home/ai/ollm-ocr-2/tita-arshia/rag
source .venv/bin/activate

# Index (use GPU when llama is stopped)
EMBED_DEVICE=cuda python3 scripts/clinic2_pipeline.py
EMBED_DEVICE=cuda python3 scripts/endo_pipeline.py

# Eval / UI
EMBED_DEVICE=cuda USE_RERANKER=1 python3 scripts/eval_test20.py
EMBED_DEVICE=cuda USE_RERANKER=1 python3 scripts/gradio_faq_app.py
```
