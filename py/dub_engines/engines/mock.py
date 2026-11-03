"""
Engine falsa: implementa todos os métodos sem nenhuma dependência de IA.
Serve para testar o protocolo e o pipeline Rust de ponta a ponta.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..audio_util import duration, read_wav, write_wav


class MockEngine:
    def __init__(self, **_: object) -> None:
        pass

    def info(self) -> dict:
        return {
            "name": "mock",
            "model": "none",
            "device": "cpu",
            "capabilities": ["transcribe", "translate", "synthesize", "separate", "diarize"],
        }

    def transcribe(self, audio: str, lang: str | None = None) -> dict:
        data, sr = read_wav(audio)
        total = duration(data, sr)
        # Um segmento falso a cada 2 s.
        segs = []
        t = 0.0
        i = 0
        while t < total:
            end = min(t + 1.6, total)
            segs.append({"start": round(t, 3), "end": round(end, 3), "text": f"mock sentence number {i}"})
            t += 2.0
            i += 1
        return {"segments": segs}

    def translate(self, texts: list[str], **_: str) -> dict:
        return {"texts": [f"frase simulada {t.split()[-1]}" if t else "" for t in texts]}

    def synthesize(self, text: str, lang: str, out_path: str, speaker_ref: str | None = None, speed: float = 1.0) -> dict:
        sr = 22050
        secs = (0.08 * len(text.split()) + 0.3) / max(speed, 0.1)
        n = int(sr * secs)
        t = np.arange(n) / sr
        tone = 0.3 * np.sin(2 * math.pi * 220.0 * t) * np.exp(-t * 1.5)
        write_wav(out_path, tone.astype(np.float32), sr)
        return {"path": out_path, "sample_rate": sr, "duration": secs}

    def separate(self, audio: str, out_dir: str) -> dict:
        data, sr = read_wav(audio)
        out = Path(out_dir)
        voice = out / "voice.wav"
        bg = out / "background.wav"
        write_wav(voice, data, sr)
        write_wav(bg, np.zeros_like(data), sr)
        return {"voice": str(voice), "background": str(bg)}

    def diarize(self, audio: str, segments: list[dict]) -> dict:
        for i, s in enumerate(segments):
            s["speaker"] = f"SPEAKER_{i % 2:02d}"
            s["overlapped"] = False
        return {"segments": segments, "speakers": ["SPEAKER_00", "SPEAKER_01"]}
