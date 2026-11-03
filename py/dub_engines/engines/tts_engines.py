"""
Engines de síntese de voz.

  piper  — ONNX, roda bem em CPU. "Pacote de voz" = arquivo .onnx + .json.
           Vozes PT-BR oficiais: pt_BR-faber-medium, pt_BR-edresson-low.
  kokoro — 82M parâmetros, voz muito natural. PT-BR: lang_code "p",
           vozes pf_dora (fem.) e pm_alex / pm_santa (masc.).
  xtts   — Coqui XTTS v2: clona a voz do falante original com ~6 s de referência.
           Pesado; GPU recomendada. Requer licença CPML (não comercial).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .. import device as dev
from ..audio_util import write_wav
from ..protocol import EngineError

log = logging.getLogger("dub.tts")


def _speed_note(speed: float) -> float:
    return float(max(0.5, min(speed or 1.0, 2.0)))


class PiperEngine:
    """`model` é o caminho do .onnx do pacote de voz (o .json fica ao lado)."""

    def __init__(self, model: str, device: str | None = None, **_: object) -> None:
        try:
            from piper import PiperVoice  # type: ignore
        except ImportError as e:
            raise EngineError("piper-tts não instalado: pip install piper-tts") from e
        p = Path(model)
        if not p.exists():
            raise EngineError(f"pacote de voz Piper não encontrado: {p}")
        self.device = dev.resolve(device)
        use_cuda = self.device == "cuda"
        log.info("carregando voz Piper %s (cuda=%s)", p.name, use_cuda)
        self.voice = PiperVoice.load(str(p), use_cuda=use_cuda)
        self.model_name = p.stem
        self.sr = int(self.voice.config.sample_rate)

    def info(self) -> dict:
        return {"name": "piper", "model": self.model_name, "device": self.device, "capabilities": ["synthesize", "speed"]}

    def synthesize(self, text: str, lang: str, out_path: str, speaker_ref: str | None = None, speed: float = 1.0) -> dict:
        speed = _speed_note(speed)
        chunks: list[np.ndarray] = []
        # API nova (>=1.3) usa SynthesisConfig; a antiga aceita kwargs. Tentamos ambas.
        try:
            from piper import SynthesisConfig  # type: ignore

            cfg = SynthesisConfig(length_scale=1.0 / speed)
            for ch in self.voice.synthesize(text, syn_config=cfg):
                chunks.append(np.frombuffer(ch.audio_int16_bytes, dtype=np.int16).astype(np.float32) / 32768.0)
        except ImportError:
            for raw in self.voice.synthesize_stream_raw(text, length_scale=1.0 / speed):
                chunks.append(np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0)
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        write_wav(out_path, audio, self.sr)
        return {"path": out_path, "sample_rate": self.sr, "duration": len(audio) / self.sr}


class KokoroEngine:
    def __init__(self, model: str = "pf_dora", device: str | None = None, **_: object) -> None:
        try:
            from kokoro import KPipeline  # type: ignore
        except ImportError as e:
            raise EngineError("kokoro não instalado: pip install kokoro soundfile") from e
        self.device = dev.resolve(device)
        self.voice = model
        self._pipes: dict[str, object] = {}
        self.sr = 24000
        self._KPipeline = KPipeline

    def _pipe(self, lang: str):
        code = {"pt": "p", "en": "a", "es": "e", "fr": "f", "it": "i", "ja": "j", "zh": "z"}.get(lang.split("-")[0].lower(), "p")
        if code not in self._pipes:
            log.info("carregando Kokoro lang=%s", code)
            self._pipes[code] = self._KPipeline(lang_code=code, device=self.device)
        return self._pipes[code]

    def info(self) -> dict:
        return {"name": "kokoro", "model": self.voice, "device": self.device, "capabilities": ["synthesize", "speed"]}

    def synthesize(self, text: str, lang: str, out_path: str, speaker_ref: str | None = None, speed: float = 1.0) -> dict:
        pipe = self._pipe(lang)
        parts = []
        for _gs, _ps, audio in pipe(text, voice=self.voice, speed=_speed_note(speed)):
            parts.append(np.asarray(audio, dtype=np.float32))
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        write_wav(out_path, audio, self.sr)
        return {"path": out_path, "sample_rate": self.sr, "duration": len(audio) / self.sr}


class XttsEngine:
    def __init__(self, model: str = "tts_models/multilingual/multi-dataset/xtts_v2", device: str | None = None, default_ref: str | None = None, **_: object) -> None:
        try:
            from TTS.api import TTS  # type: ignore
        except ImportError as e:
            raise EngineError("coqui-tts não instalado: pip install coqui-tts") from e
        self.device = dev.resolve(device)
        self.model_name = model
        self.default_ref = default_ref
        log.info("carregando XTTS em %s (pode demorar)", self.device)
        self.tts = TTS(model).to(self.device)
        self.sr = 24000

    def info(self) -> dict:
        return {"name": "xtts", "model": self.model_name, "device": self.device, "capabilities": ["synthesize", "clone", "speed"]}

    def synthesize(self, text: str, lang: str, out_path: str, speaker_ref: str | None = None, speed: float = 1.0) -> dict:
        ref = speaker_ref or self.default_ref
        if not ref or not Path(ref).exists():
            raise EngineError("XTTS precisa de um WAV de referência (speaker_ref)")
        code = lang.split("-")[0].lower()
        wav = self.tts.tts(text=text, speaker_wav=ref, language=code, speed=_speed_note(speed))
        audio = np.asarray(wav, dtype=np.float32)
        write_wav(out_path, audio, self.sr)
        return {"path": out_path, "sample_rate": self.sr, "duration": len(audio) / self.sr}
