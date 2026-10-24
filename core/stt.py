"""Speech-to-text with Faster-Whisper (Arabic + English, language auto-detected)."""
from __future__ import annotations

import os
import tempfile
import threading
from dataclasses import dataclass

from . import config

_model = None
_lock = threading.Lock()


@dataclass
class Transcript:
    text: str
    language: str
    language_probability: float
    duration: float


def get_model():
    """Load the Whisper model once (lazy, thread-safe)."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from faster_whisper import WhisperModel
                _model = WhisperModel(config.WHISPER_MODEL, device=config.WHISPER_DEVICE,
                                      compute_type=config.WHISPER_COMPUTE_TYPE)
    return _model


def transcribe_file(path: str, language: str | None = None) -> Transcript:
    """language: 'ar' / 'en' to force, or None to auto-detect."""
    model = get_model()
    segments, info = model.transcribe(path, language=language or None, vad_filter=True, beam_size=5,
                                      initial_prompt=None)
    text = " ".join(s.text.strip() for s in segments).strip()
    return Transcript(text=text, language=info.language, language_probability=float(info.language_probability),
                      duration=float(info.duration))


def transcribe_bytes(data: bytes, filename: str = "audio.wav", language: str | None = None) -> Transcript:
    suffix = os.path.splitext(filename)[1] or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return transcribe_file(path, language)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
