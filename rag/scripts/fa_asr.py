"""Persian speech-to-text with Vosk (alphacep/vosk-api).

Default model is the Farsi pack vosk-model-fa-0.42. Audio is converted to
16 kHz mono PCM before recognition. Vosk runs on CPU.
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
from vosk import KaldiRecognizer, Model

SAMPLE_RATE = 16000
_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "vosk-model-fa-0.42"
MODEL_DIR = os.environ.get("VOSK_MODEL", str(_MODEL_DIR))

_model: Model | None = None
_lock = threading.Lock()


def get_model() -> Model:
    global _model
    if _model is None:
        path = Path(MODEL_DIR)
        if not path.is_dir():
            raise FileNotFoundError(f"Vosk model not found: {path}")
        _model = Model(str(path))
    return _model


def _to_pcm16(data: bytes) -> bytes:
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
    if sr != SAMPLE_RATE:
        wav = torch.from_numpy(mono).unsqueeze(0)
        mono = torchaudio.functional.resample(wav, int(sr), SAMPLE_RATE).squeeze(0).numpy()
    pcm = np.clip(mono, -1.0, 1.0)
    return (pcm * 32767.0).astype(np.int16).tobytes()


def transcribe(data: bytes) -> str:
    pcm = _to_pcm16(data)
    with _lock:
        rec = KaldiRecognizer(get_model(), SAMPLE_RATE)
        step = 8000
        for i in range(0, len(pcm), step):
            rec.AcceptWaveform(pcm[i : i + step])
        result = json.loads(rec.FinalResult())
    return str(result.get("text", "")).strip()
