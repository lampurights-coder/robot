#!/usr/bin/env python3
"""Simple Gradio + LangGraph FAQ assistant (Clinic2 only).

Flow:
  user question → Jina embed → Qdrant top-K
               → BAAI/bge-reranker-v2-m3 rerank
               → if below threshold: refuse
               → else: local llama.cpp answers ONLY from retrieved FAQ answer
"""

from __future__ import annotations

import os
from typing import TypedDict

import gradio as gr
import httpx
import torch
from langgraph.graph import END, START, StateGraph
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- config ---
LLM_BASE = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen3.8-27B")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "clinic2_faq_qa")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "jinaai/jina-embeddings-v5-text-small")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
USE_RERANKER = os.environ.get("USE_RERANKER", "1") not in ("0", "false", "False")
SCORE_MIN = float(os.environ.get("SCORE_MIN", "0.55"))
RERANK_MIN = float(os.environ.get("RERANK_MIN", "0.25"))
QDRANT_FLOOR = float(os.environ.get("QDRANT_FLOOR", "0.30"))
CANDIDATE_K = int(os.environ.get("CANDIDATE_K", "10"))
TOP_K = int(os.environ.get("TOP_K", "3"))
HOST = os.environ.get("GRADIO_HOST", "0.0.0.0")
PORT = int(os.environ.get("GRADIO_PORT", "7860"))

EMBED_DEVICE = os.environ.get(
    "EMBED_DEVICE",
    "cpu"
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == ""
    else ("cuda" if torch.cuda.is_available() else "cpu"),
)


class State(TypedDict, total=False):
    question: str
    score: float
    qdrant_score: float
    rerank_score: float
    card_id: str
    matched_question: str
    faq_answer: str
    context: str
    reply: str
    refused: bool


class BgeReranker:
    def __init__(self, model_id: str, device: str = "cpu"):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.bfloat16 if device.startswith("cuda") else None
        kwargs = {"dtype": dtype} if dtype is not None else {}
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
    q = (payload.get("question") or "").strip()
    a = (payload.get("answer") or "").strip()
    if a:
        return f"{q}\n{a[:400]}"
    return q


print(f"Loading embedder {EMBED_MODEL} on {EMBED_DEVICE}...")
_embed_kwargs = {"trust_remote_code": True, "device": EMBED_DEVICE}
if EMBED_DEVICE == "cuda":
    _embed_kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
embedder = SentenceTransformer(EMBED_MODEL, **_embed_kwargs)
qdrant = QdrantClient(url=QDRANT_URL, check_compatibility=False)

reranker = None
if USE_RERANKER:
    print(f"Loading reranker {RERANK_MODEL} on {EMBED_DEVICE}...")
    reranker = BgeReranker(RERANK_MODEL, device=EMBED_DEVICE)


def retrieve(state: State) -> State:
    q = state["question"].strip()
    vec = embedder.encode(
        [q],
        task="retrieval",
        prompt_name="query",
        normalize_embeddings=True,
    )[0]
    res = qdrant.query_points(
        collection_name=COLLECTION,
        query=vec.tolist(),
        limit=CANDIDATE_K if USE_RERANKER else TOP_K,
        with_payload=True,
    )
    hits = list(res.points)
    refuse_msg = (
        "در این FAQ پاسخی برای این سؤال پیدا نشد. "
        "لطفاً سؤال را واضح‌تر بپرسید یا با کلینیک تماس بگیرید."
    )

    if not hits:
        return {
            **state,
            "refused": True,
            "score": 0.0,
            "qdrant_score": 0.0,
            "rerank_score": 0.0,
            "card_id": "",
            "matched_question": "",
            "faq_answer": "",
            "context": "",
            "reply": refuse_msg,
        }

    if USE_RERANKER and reranker is not None:
        usable = [h for h in hits if ((h.payload or {}).get("answer") or "").strip()]
        if not usable:
            usable = hits
        docs = [candidate_text(h.payload or {}) for h in usable]
        scores = reranker.score(q, docs)
        best_by_card: dict[str, tuple[float, int]] = {}
        for i, (h, s) in enumerate(zip(usable, scores)):
            cid = str((h.payload or {}).get("id") or i)
            if cid not in best_by_card or s > best_by_card[cid][0]:
                best_by_card[cid] = (float(s), i)
        best_i = max(best_by_card.values(), key=lambda t: t[0])[1]
        top = usable[best_i]
        rerank_s = float(scores[best_i])
        qdrant_s = float(top.score)
        accept = (rerank_s >= RERANK_MIN and qdrant_s >= QDRANT_FLOOR) or (
            qdrant_s >= SCORE_MIN
        )
        display_score = rerank_s
        # context: top reranked cards (unique)
        ranked = sorted(
            zip(usable, scores), key=lambda t: t[1], reverse=True
        )
    else:
        top = hits[0]
        rerank_s = 0.0
        qdrant_s = float(top.score)
        accept = qdrant_s >= SCORE_MIN
        display_score = qdrant_s
        ranked = [(h, h.score) for h in hits]

    if not accept:
        return {
            **state,
            "refused": True,
            "score": display_score,
            "qdrant_score": qdrant_s,
            "rerank_score": rerank_s,
            "card_id": (top.payload or {}).get("id", ""),
            "matched_question": (top.payload or {}).get("question", ""),
            "faq_answer": "",
            "context": "",
            "reply": refuse_msg,
        }

    lines = []
    seen = set()
    for h, s in ranked:
        if USE_RERANKER:
            if float(s) < RERANK_MIN and float(h.score) < SCORE_MIN:
                continue
        elif float(h.score) < SCORE_MIN:
            continue
        cid = (h.payload or {}).get("id")
        if cid in seen:
            continue
        seen.add(cid)
        lines.append(
            f"[کارت {cid}]\n"
            f"سؤال‌های مرتبط: {' | '.join((h.payload or {}).get('all_questions') or [(h.payload or {}).get('question')])}\n"
            f"پاسخ رسمی: {(h.payload or {}).get('answer')}"
        )
        if len(seen) >= TOP_K:
            break

    return {
        **state,
        "refused": False,
        "score": display_score,
        "qdrant_score": qdrant_s,
        "rerank_score": rerank_s,
        "card_id": (top.payload or {}).get("id", ""),
        "matched_question": (top.payload or {}).get("question", ""),
        "faq_answer": (top.payload or {}).get("answer", ""),
        "context": "\n\n".join(lines),
    }


def generate(state: State) -> State:
    if state.get("refused"):
        return state

    system = (
        "تو دستیار پاسخ‌گویی FAQ کلینیک دندانپزشکی هستی.\n"
        "فقط و فقط بر اساس «متن پاسخ رسمی» که در زمینه آمده جواب بده.\n"
        "هیچ واقعیت پزشکی یا عددی جدیدی نساز.\n"
        "اگر زمینه کافی نیست بگو نمی‌دانی.\n"
        "مختصر، مؤدب و به فارسی بنویس."
    )
    # Keep under llama -c 8192 (long implant cards can be 20k+ chars)
    faq_limit = int(os.environ.get("LLM_FAQ_CHARS", "10000"))
    context = state.get("context") or ""
    if len(context) > faq_limit:
        context = context[:faq_limit].rstrip() + "\n…"
    user = (
        f"سؤال بیمار: {state['question']}\n\n"
        f"زمینه FAQ (فقط از این استفاده کن):\n{context}"
    )
    try:
        r = httpx.post(
            f"{LLM_BASE}/chat/completions",
            json={
                "model": LLM_MODEL,
                "temperature": 0.3,
                "top_p": 0.8,
                "max_tokens": 500,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=180.0,
        )
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        reply = (msg.get("content") or "").strip()
        if not reply:
            reply = state.get("faq_answer") or state["reply"]
    except Exception as e:
        reply = state.get("faq_answer") or f"خطا در مدل: {e}"

    return {**state, "reply": reply}


def build_graph():
    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


graph = build_graph()


def chat(message: str, history: list):
    if not message or not message.strip():
        return history, ""
    out = graph.invoke({"question": message.strip()})
    reply = out.get("reply", "")
    meta = (
        f"\n\n—\n"
        f"card=`{out.get('card_id')}`  "
        f"qdrant={out.get('qdrant_score', 0):.3f}  "
        f"rerank={out.get('rerank_score', 0):.3f}  "
        f"matched=`{out.get('matched_question', '')}`"
    )
    if out.get("refused"):
        full = reply
    else:
        full = reply + meta
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": full},
    ]
    return history, ""


def build_ui():
    with gr.Blocks(title="Clinic2 FAQ RAG", fill_height=True) as demo:
        rerank_label = RERANK_MODEL if USE_RERANKER else "off"
        gr.Markdown(
            "# FAQ کلینیک (Clinic2)\n"
            "پاسخ **فقط** از کارت‌های FAQ بازیابی‌شده. "
            f"آستانه dense: `{SCORE_MIN}` · rerank: `{rerank_label}` · "
            f"مدل: `{LLM_MODEL}` · embedding: Jina v5-small"
        )
        chatbot = gr.Chatbot(height=480, rtl=True)
        with gr.Row():
            msg = gr.Textbox(
                placeholder="مثال: هزینه روکش چقدره؟",
                scale=5,
                show_label=False,
                rtl=True,
            )
            send = gr.Button("بپرس", variant="primary", scale=1)
        gr.Examples(
            examples=[
                "هزینه روکش چقدره؟",
                "هزینه ایمپلنت چقدره؟",
                "جرمگیری چقدر هزینه داره؟",
                "جانبازا خدمات رایگان دارن؟",
                "روکشم افتاد باید وقت بگیرم؟",
            ],
            inputs=msg,
        )

        send.click(chat, [msg, chatbot], [chatbot, msg])
        msg.submit(chat, [msg, chatbot], [chatbot, msg])
    return demo


if __name__ == "__main__":
    ui = build_ui()
    print(f"Gradio → http://{HOST}:{PORT}")
    ui.launch(server_name=HOST, server_port=PORT, share=False)
