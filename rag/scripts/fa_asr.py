"""Persian speech-to-text with Vosk (https://github.com/alphacep/vosk-api).

Weights: vosk-model-fa-0.42 (large Persian model). Audio is resampled to
16 kHz mono PCM and recognized on CPU.
"""

from __future__ import annotations

import io
import json
import os
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from vosk import KaldiRecognizer, Model, SetLogLevel

SetLogLevel(-1)

_DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "vosk-fa"
MODEL_DIR = Path(os.environ.get("VOSK_MODEL", str(_DEFAULT_MODEL)))
SAMPLE_RATE = 16000

_model: Model | None = None
_lock = threading.Lock()


def get_model() -> Model:
    global _model
    if _model is None:
        if not (MODEL_DIR / "am" / "final.mdl").is_file():
            raise FileNotFoundError(f"Vosk model not found: {MODEL_DIR}")
        _model = Model(str(MODEL_DIR))
    return _model


def _to_pcm16(data: bytes) -> bytes:
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
    if int(sr) != SAMPLE_RATE and mono.size:
        wav = torch.from_numpy(mono).unsqueeze(0)
        mono = (
            torchaudio.functional.resample(wav, int(sr), SAMPLE_RATE)
            .squeeze(0)
            .numpy()
        )
    pcm = np.clip(mono, -1.0, 1.0)
    return (pcm * 32767.0).astype(np.int16).tobytes()


def transcribe(data: bytes) -> str:
    pcm = _to_pcm16(data)
    if not pcm:
        return ""
    model = get_model()
    with _lock:
        rec = KaldiRecognizer(model, SAMPLE_RATE)
        step = 8000
        for i in range(0, len(pcm), step):
            rec.AcceptWaveform(pcm[i : i + step])
        result = json.loads(rec.FinalResult())
    return str(result.get("text") or "").strip()
