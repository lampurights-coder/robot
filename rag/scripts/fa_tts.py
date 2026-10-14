"""Persian Chatterbox TTS.

Loads ResembleAI multilingual Chatterbox from the local HF cache, then
replaces the T3 weights with Thomcles/Chatterbox-TTS-Persian-Farsi
(`t3_fa.safetensors`). Runs on GPU.
"""

from __future__ import annotations

import io
import os
import re
import threading
from pathlib import Path

import numpy as np
import torch

CKPT_DIR = Path(
    os.environ.get(
        "CHATTERBOX_CKPT",
        "/home/ai/.cache/huggingface/hub/models--ResembleAI--chatterbox/"
        "snapshots/05e904af2b5c7f8e482687a9d7336c5c824467d9",
    )
)
T3_FA = Path(
    os.environ.get(
        "T3_FA_WEIGHTS",
        "/home/ai/ollm-ocr-2/tita-arshia/t3_fa.safetensors",
    )
)
TTS_DEVICE = os.environ.get("TTS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
# One Chatterbox pass stops at 1000 speech tokens (~40s). Keep each piece shorter.
TTS_CHUNK_CHARS = int(os.environ.get("TTS_CHUNK_CHARS", "180"))
TTS_PAUSE_SEC = float(os.environ.get("TTS_PAUSE_SEC", "0.2"))

_model = None
_lock = threading.Lock()
_SPLIT = re.compile(r"(?<=[\.!\?؟\n])\s+")


def split_for_speech(text: str, limit: int = TTS_CHUNK_CHARS) -> list[str]:
    """Split on sentence ends, then pack pieces that still fit in one pass."""
    text = " ".join((text or "").split())
    if not text:
        return []
    sentences = [p.strip() for p in _SPLIT.split(text) if p.strip()]
    chunks: list[str] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf:
            chunks.append(buf)
            buf = ""

    for sentence in sentences:
        while len(sentence) > limit:
            flush()
            chunks.append(sentence[:limit].strip())
            sentence = sentence[limit:].strip()
        if not sentence:
            continue
        if buf and len(buf) + 1 + len(sentence) > limit:
            flush()
        buf = f"{buf} {sentence}".strip()
    flush()
    return chunks


def get_model():
    global _model
    if _model is not None:
        return _model
    if not CKPT_DIR.is_dir():
        raise FileNotFoundError(f"Chatterbox checkpoint dir missing: {CKPT_DIR}")
    if not T3_FA.is_file():
        raise FileNotFoundError(f"Persian T3 weights missing: {T3_FA}")

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from safetensors.torch import load_file

    print(f"Loading Chatterbox multilingual from {CKPT_DIR} on {TTS_DEVICE}...")
    model = ChatterboxMultilingualTTS.from_local(CKPT_DIR, TTS_DEVICE)
    print(f"Loading Persian T3 {T3_FA}...")
    state = load_file(str(T3_FA), device="cpu")
    model.t3.load_state_dict(state)
    model.t3.to(TTS_DEVICE).eval()
    _model = model
    print("Persian TTS ready.")
    return _model


def synthesize(text: str) -> tuple[bytes, str, int]:
    """Return (wav_bytes, spoken_text, sample_rate). Long text is spoken in full."""
    chunks = split_for_speech(text)
    if not chunks:
        raise ValueError("nothing to speak")
    pieces: list[np.ndarray] = []
    with _lock:
        model = get_model()
        sr = int(model.sr)
        pause = np.zeros(int(sr * TTS_PAUSE_SEC), dtype=np.float32)
        for i, chunk in enumerate(chunks, start=1):
            print(f"TTS chunk {i}/{len(chunks)} ({len(chunk)} chars)")
            with torch.inference_mode():
                wav = model.generate(chunk, language_id=None)
            audio = wav.squeeze().detach().float().cpu().numpy()
            pieces.append(audio)
            if i != len(chunks):
                pieces.append(pause)
    audio = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
    # 16-bit PCM is what browsers can play. Float WAV files open and then stay silent.
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    import soundfile as sf

    sf.write(buf, pcm, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue(), " ".join(chunks), sr
