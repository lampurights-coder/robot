#!/usr/bin/env python3
"""FastAPI FAQ RAG — same pipeline as gradio_faq_app.py.

  question → Jina embed → Qdrant top-K → BGE rerank → refuse or Qwen answer

Run:
  EMBED_DEVICE=cuda USE_RERANKER=1 python3 scripts/fastapi_faq_app.py
  # or: uvicorn scripts.fastapi_faq_app:app --host 0.0.0.0 --port 7860

Ask:
  curl -s http://127.0.0.1:7860/ask -H 'Content-Type: application/json' \\
    -d '{"question":"هزینه روکش چقدره؟"}' | jq .
"""

from __future__ import annotations

import os
import re
import secrets
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# so `from weather_mcp_client import ...` works when launched as scripts/fastapi_faq_app.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
LLM_FAQ_CHARS = int(os.environ.get("LLM_FAQ_CHARS", "10000"))
HOST = os.environ.get("API_HOST", "0.0.0.0")
PORT = int(os.environ.get("API_PORT", "7860"))

EMBED_DEVICE = os.environ.get(
    "EMBED_DEVICE",
    "cpu"
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == ""
    else ("cuda" if torch.cuda.is_available() else "cpu"),
)

REFUSE_MSG = (
    "در این FAQ پاسخی برای این سؤال پیدا نشد. "
    "لطفاً سؤال را واضح‌تر بپرسید یا با کلینیک تماس بگیرید."
)


class BgeReranker:
    def __init__(self, model_id: str, device: str = "cpu"):
        self.device = device
        offline = os.environ.get("HF_HUB_OFFLINE", "0") in ("1", "true", "True")
        self.tok = AutoTokenizer.from_pretrained(model_id, local_files_only=offline)
        dtype = torch.bfloat16 if device.startswith("cuda") else None
        kwargs: dict[str, Any] = {"local_files_only": offline}
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
    q = (payload.get("question") or "").strip()
    a = (payload.get("answer") or "").strip()
    if a:
        return f"{q}\n{a[:400]}"
    return q


# loaded in lifespan
embedder: SentenceTransformer | None = None
qdrant: QdrantClient | None = None
reranker: BgeReranker | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global embedder, qdrant, reranker
    print(f"Loading embedder {EMBED_MODEL} on {EMBED_DEVICE}...")
    kwargs: dict[str, Any] = {"trust_remote_code": True, "device": EMBED_DEVICE}
    if EMBED_DEVICE == "cuda":
        kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
    embedder = SentenceTransformer(EMBED_MODEL, **kwargs)
    qdrant = QdrantClient(url=QDRANT_URL, check_compatibility=False)
    if USE_RERANKER:
        print(f"Loading reranker {RERANK_MODEL} on {EMBED_DEVICE}...")
        reranker = BgeReranker(RERANK_MODEL, device=EMBED_DEVICE)
    yield


app = FastAPI(
    title="Clinic FAQ RAG",
    description="Jina → Qdrant → BGE rerank → local Qwen (FAQ-grounded answers)",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Length", "Content-Type", "X-Voice-Url"],
)

# each POST /speak writes its own file under data/voices/
VOICES_DIR = Path(__file__).resolve().parents[1] / "data" / "voices"
LAST_NAME = ""
_VOICE_NAME = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{4}\.wav$")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["هزینه روکش چقدره؟"])
    skip_llm: bool = Field(
        False,
        description="If true, return raw FAQ answer without calling Qwen",
    )


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["سلام. هزینه روکش بعد از معاینه مشخص می‌شود."])


class AsrResponse(BaseModel):
    text: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    card_id: str = ""
    matched_question: str = ""
    faq_answer: str = ""
    qdrant_score: float = 0.0
    rerank_score: float = 0.0
    score: float = 0.0
    source: str = "faq"  # faq | weather_mcp | ecotrust
    weather_tool: str = ""


def retrieve(question: str) -> dict[str, Any]:
    assert embedder is not None and qdrant is not None
    q = question.strip()
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
    empty = {
        "refused": True,
        "score": 0.0,
        "qdrant_score": 0.0,
        "rerank_score": 0.0,
        "card_id": "",
        "matched_question": "",
        "faq_answer": "",
        "context": "",
        "reply": REFUSE_MSG,
    }
    if not hits:
        return empty

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
        ranked = sorted(zip(usable, scores), key=lambda t: t[1], reverse=True)
    else:
        top = hits[0]
        rerank_s = 0.0
        qdrant_s = float(top.score)
        accept = qdrant_s >= SCORE_MIN
        display_score = qdrant_s
        ranked = [(h, h.score) for h in hits]

    if not accept:
        return {
            "refused": True,
            "score": display_score,
            "qdrant_score": qdrant_s,
            "rerank_score": rerank_s,
            "card_id": (top.payload or {}).get("id", "") or "",
            "matched_question": (top.payload or {}).get("question", "") or "",
            "faq_answer": "",
            "context": "",
            "reply": REFUSE_MSG,
        }

    lines: list[str] = []
    seen: set[Any] = set()
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
        "refused": False,
        "score": display_score,
        "qdrant_score": qdrant_s,
        "rerank_score": rerank_s,
        "card_id": (top.payload or {}).get("id", "") or "",
        "matched_question": (top.payload or {}).get("question", "") or "",
        "faq_answer": (top.payload or {}).get("answer", "") or "",
        "context": "\n\n".join(lines),
        "reply": "",
    }


def generate(question: str, state: dict[str, Any]) -> str:
    if state.get("refused"):
        return state["reply"]

    system = (
        "تو دستیار پاسخ‌گویی FAQ کلینیک دندانپزشکی هستی.\n"
        "فقط و فقط بر اساس «متن پاسخ رسمی» که در زمینه آمده جواب بده.\n"
        "هیچ واقعیت پزشکی یا عددی جدیدی نساز.\n"
        "اگر زمینه کافی نیست بگو نمی‌دانی.\n"
        "مختصر، مؤدب و به فارسی بنویس."
    )
    context = state.get("context") or ""
    if len(context) > LLM_FAQ_CHARS:
        context = context[:LLM_FAQ_CHARS].rstrip() + "\n…"
    user = f"سؤال بیمار: {question}\n\nزمینه FAQ (فقط از این استفاده کن):\n{context}"
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
            reply = state.get("faq_answer") or REFUSE_MSG
    except Exception as e:
        reply = state.get("faq_answer") or f"خطا در مدل: {e}"
    return reply


@app.get("/health")
def health():
    return {
        "status": "ok",
        "embed_device": EMBED_DEVICE,
        "use_reranker": USE_RERANKER,
        "collection": COLLECTION,
        "llm": LLM_MODEL,
    }


def answer_weather(question: str) -> AskResponse:
    """Weather city questions → local tita-arshia/agent weather_mcp tool (no email)."""
    from weather_mcp_client import call_weather, guess_location

    location = guess_location(question)
    try:
        got = call_weather(location)
    except Exception as e:
        raise HTTPException(502, f"Weather tool failed: {e}") from e

    raw = got["text"]
    return AskResponse(
        answer=raw,
        refused=False,
        card_id="",
        matched_question=f"weather:{location}",
        faq_answer=raw[:2000],
        qdrant_score=0.0,
        rerank_score=0.0,
        score=1.0,
        source="weather_mcp",
        weather_tool=str(got.get("tool") or "get_weather"),
    )


def answer_dollar_rial(question: str) -> AskResponse:
    """Dollar / gold / coin prices → scrape https://ecotrust.ir/ (no web search)."""
    from dollar_rial_client import call_usd_irr

    try:
        got = call_usd_irr(question)
    except Exception as e:
        raise HTTPException(502, f"Ecotrust price tool failed: {e}") from e

    raw = got["text"]
    symbol = (got.get("raw") or {}).get("symbol") or "ecotrust"
    return AskResponse(
        answer=raw,
        refused=False,
        card_id="",
        matched_question=f"ecotrust:{symbol}",
        faq_answer=raw[:2000],
        qdrant_score=0.0,
        rerank_score=0.0,
        score=1.0,
        source="ecotrust",
        weather_tool=str(got.get("tool") or "ecotrust_get_market_price"),
    )


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest):
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "question is empty")

    # Tools first (no email); clinic FAQ otherwise
    from dollar_rial_client import is_dollar_rial_question
    from weather_mcp_client import is_weather_question

    if is_dollar_rial_question(q):
        return answer_dollar_rial(q)
    if is_weather_question(q):
        return answer_weather(q)

    if embedder is None or qdrant is None:
        raise HTTPException(503, "models not loaded yet")

    state = retrieve(q)
    if state["refused"]:
        answer = REFUSE_MSG
    elif body.skip_llm:
        answer = state["faq_answer"] or REFUSE_MSG
    else:
        answer = generate(q, state)

    return AskResponse(
        answer=answer,
        refused=bool(state["refused"]),
        card_id=str(state.get("card_id") or ""),
        matched_question=str(state.get("matched_question") or ""),
        faq_answer=str(state.get("faq_answer") or "")[:2000],
        qdrant_score=float(state.get("qdrant_score") or 0),
        rerank_score=float(state.get("rerank_score") or 0),
        score=float(state.get("score") or 0),
        source="faq",
    )


def _new_voice_name() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(2)}.wav"


def _save_voice(wav: bytes) -> str:
    global LAST_NAME
    name = _new_voice_name()
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    (VOICES_DIR / name).write_bytes(wav)
    LAST_NAME = name
    return name


def _wav_response(wav: bytes, name: str) -> Response:
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'inline; filename="{name}"',
            "X-Voice-Url": f"/voices/{name}",
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-store",
        },
    )


@app.post("/speak")
def speak(body: SpeakRequest):
    """Turn text into a Persian wav. Model runs on GPU."""
    try:
        from fa_tts import synthesize

        wav, _spoken, _sr = synthesize(body.text)
    except Exception as exc:
        raise HTTPException(500, f"TTS failed: {type(exc).__name__}: {exc}") from exc
    name = _save_voice(wav)
    return _wav_response(wav, name)


@app.post("/asr", response_model=AsrResponse)
async def asr(file: UploadFile = File(...)):
    """Persian speech to text. Send a wav as multipart `file`."""
    data = await file.read()
    if not data:
        raise HTTPException(400, "audio file is empty")
    try:
        from fa_asr import transcribe

        text = transcribe(data)
    except Exception as exc:
        raise HTTPException(500, f"ASR failed: {type(exc).__name__}: {exc}") from exc
    return AsrResponse(text=text)


@app.get("/voices/{name}")
def voice_file(name: str):
    """Play one saved clip. Each speak call has its own name."""
    if not _VOICE_NAME.match(name):
        raise HTTPException(404, "unknown voice")
    path = VOICES_DIR / name
    if not path.is_file():
        raise HTTPException(404, "voice not found")
    return FileResponse(
        path,
        media_type="audio/wav",
        filename=name,
        content_disposition_type="inline",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/voice.wav")
@app.get("/speak")
@app.get("/audio/last.wav")
def latest_voice():
    """Old address: send the browser to the latest uniquely named file."""
    if not LAST_NAME:
        raise HTTPException(404, "no audio yet — POST /speak with text first")
    return RedirectResponse(url=f"/voices/{LAST_NAME}", status_code=307)


@app.get("/voice", response_class=HTMLResponse)
def voice_page(request: Request):
    """Page with a player, for browsers that will not play a bare wav link."""
    if not LAST_NAME:
        raise HTTPException(404, "no audio yet — POST /speak with text first")
    src = f"/voices/{LAST_NAME}"
    wav_url = str(request.base_url).rstrip("/") + src
    return f"""<!doctype html>
<html lang="fa">
<head>
  <meta charset="utf-8"/>
  <title>{LAST_NAME}</title>
</head>
<body>
  <audio id="player" controls autoplay src="{src}"></audio>
  <p>{wav_url}</p>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def player_page():
    return """<!doctype html>
<html lang="fa" dir="rtl">
<head>
  <meta charset="utf-8"/>
  <title>Speak</title>
  <style>
    body { font-family: sans-serif; max-width: 640px; margin: 2rem auto; }
    textarea { width: 100%; min-height: 6rem; font-size: 1.1rem; }
    button { font-size: 1.05rem; padding: 0.4rem 1rem; }
    audio { width: 100%; margin-top: 1rem; }
  </style>
</head>
<body>
  <h1>پخش صدا</h1>
  <textarea id="text">سلام. این یک آزمایش پخش صدا است.</textarea>
  <p><button id="play" type="button">Play</button></p>
  <p id="status"></p>
  <audio id="player" controls></audio>
  <script>
    const btn = document.getElementById("play");
    const player = document.getElementById("player");
    btn.onclick = async () => {
      btn.disabled = true;
      document.getElementById("status").textContent = "در حال ساخت صدا…";
      try {
        const r = await fetch("/speak", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({text: document.getElementById("text").value})
        });
        if (!r.ok) {
          const err = await r.json().catch(() => ({}));
          throw new Error(err.detail || r.statusText);
        }
        const fileUrl = r.headers.get("X-Voice-Url") || "";
        const blob = await r.blob();
        player.src = fileUrl || URL.createObjectURL(blob);
        await player.play();
        document.getElementById("status").innerHTML = fileUrl
          ? 'در حال پخش — <a href="' + fileUrl + '">' + fileUrl.split("/").pop() + "</a>"
          : "در حال پخش";
      } catch (err) {
        document.getElementById("status").textContent = String(err);
      } finally {
        btn.disabled = false;
      }
    };
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    print(f"FastAPI → http://{HOST}:{PORT}  (GET /health, POST /ask, POST /speak, POST /asr)")
    uvicorn.run(app, host=HOST, port=PORT)
