# Tita Arshia

Persian dental-clinic assistant. A question is answered from the clinic FAQ, or from a live tool when it is about weather or market prices. The same service can speak an answer and transcribe a voice clip.

The API listens on port **7860**. From another machine on the LAN, use this host’s address, not `127.0.0.1`.

## What it does

`POST /ask` routes in this order:

1. **Prices** — dollar, tether, gold, coin, and listed FX from [ecotrust.ir](https://ecotrust.ir/). Toman values are also shown in Rial (×10). No web search.
2. **Weather** — city weather from the local tool in `agent/weather_mcp`. No email.
3. **Clinic FAQ** — Jina embeddings, Qdrant, then BGE rerank. A weak match is refused. A strong match is rewritten by the local LLM from the FAQ card only.

`skip_llm: true` returns the stored FAQ answer without calling the LLM.

Speech:

- `POST /speak` — Persian text in, WAV out (Chatterbox with Persian T3 weights, GPU).
- `GET /voices/<name>.wav` — play that clip in a browser. Each call gets its own filename.
- `POST /asr` — audio file in, Persian text out ([Vosk](https://github.com/alphacep/vosk-api) Farsi model `vosk-model-fa-0.42`, CPU).

## Layout

```
tita-arshia/
  faqs/sources/     numbered FAQ text and HTML (endo, clinic2, anesthesia, …)
  faqs/eval/        small eval set
  rag/scripts/      ingest, eval, FastAPI, TTS, ASR
  rag/data/         parsed FAQ cards (not the vector index)
  agent/            weather tool and ecotrust price scraper
  infrence.py       example client: POST /ask and write answer.json
```

Qdrant collection: `clinic2_faq_qa`. Each stored point is one FAQ question. Several questions can share one answer. `matched_question` is the stored question, not the text the user typed.

## Run

Needs Docker, a CUDA GPU for embed/rerank/TTS, and a llama.cpp server on port 8080 for FAQ answers that call the LLM. This repo does not include that server or any weights.

```bash
cd rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

./scripts/start_stack.sh
# also start the LLM on :8080:
# START_LLM=1 ./scripts/start_stack.sh
```

Stop the API and the LLM with `./scripts/stop_stack.sh`. Qdrant stays up unless `STOP_QDRANT=1`.

Ingest one FAQ source (Qdrant must be up):

```bash
./scripts/ingest_faq.sh restorative
./scripts/ingest_faq.sh "FAQ-General Anesthesia"
```

Eval (retrieval and end-to-end answer quality):

```bash
./scripts/run_all_evals.sh
```

If embed and the LLM do not fit on the GPU together, run eval with `EMBED_DEVICE=cpu`.

## API

Health:

```bash
curl -s http://127.0.0.1:7860/health
```

Ask:

```bash
curl -s http://127.0.0.1:7860/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"هزینه روکش چقدره؟"}'
```

Speak:

```bash
curl -s http://127.0.0.1:7860/speak \
  -H 'Content-Type: application/json' \
  -d '{"text":"سلام. هزینه روکش بعد از معاینه مشخص می‌شود."}' \
  -D - -o reply.wav
```

The response header `X-Voice-Url` is the browser URL for that file.

Transcribe:

```bash
curl -s -F "file=@speech.wav" http://127.0.0.1:7860/asr
```

`GET /` is a small page that plays synthesized speech.

## Weights (not in git)

Download these next to the code. They are listed in `.gitignore`.

| Piece | Where it goes |
| --- | --- |
| [jinaai/jina-embeddings-v5-text-small](https://huggingface.co/jinaai/jina-embeddings-v5-text-small) | Hugging Face cache (`rag/.hf_cache` when `HF_HOME` is set by `start_stack.sh`) |
| [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | same cache |
| [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) | `CHATTERBOX_CKPT` |
| [Thomcles/Chatterbox-TTS-Persian-Farsi](https://huggingface.co/Thomcles/Chatterbox-TTS-Persian-Farsi) `t3_fa.safetensors` | `T3_FA_WEIGHTS` (default `tita-arshia/t3_fa.safetensors`) |
| [vosk-model-fa-0.42](https://alphacephei.com/vosk/models) | `rag/models/vosk-model-fa-0.42` or `VOSK_MODEL` |
| LLM GGUF served as `Qwen3.8-27B` on `:8080` | outside this folder |

The checked-in client name for the LLM is `Qwen3.8-27B`. Point llama.cpp at whichever GGUF you run, and keep that alias so `/ask` does not need a code change.
