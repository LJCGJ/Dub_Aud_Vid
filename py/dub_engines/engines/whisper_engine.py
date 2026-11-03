"""Transcrição com faster-whisper (CTranslate2). GPU se disponível."""

from __future__ import annotations

import logging

from .. import device as dev
from ..protocol import EngineError

log = logging.getLogger("dub.whisper")


class WhisperEngine:
    def __init__(self, model: str = "medium", device: str | None = None, threads: int | None = None, **_: object) -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as e:
            raise EngineError("faster-whisper não instalado: pip install faster-whisper") from e

        self.device = dev.resolve(device)
        n = dev.cpu_threads(threads)
        self.model_name = model
        log.info("carregando whisper %s em %s", model, self.device)
        self.model = WhisperModel(
            model,
            device=self.device,
            compute_type=dev.ct2_compute_type(self.device),
            cpu_threads=n,
        )

    def info(self) -> dict:
        return {"name": "faster-whisper", "model": self.model_name, "device": self.device, "capabilities": ["transcribe"]}

    def transcribe(self, audio: str, lang: str | None = None) -> dict:
        language = lang.split("-")[0] if lang else None
        segments, _info = self.model.transcribe(
            audio,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
            condition_on_previous_text=False,
            word_timestamps=False,
        )
        out = []
        for s in segments:
            text = s.text.strip()
            if text:
                out.append({"start": float(s.start), "end": float(s.end), "text": text})
        return {"segments": out}
