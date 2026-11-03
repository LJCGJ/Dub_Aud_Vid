"""Separação voz/fundo (Demucs) e diarização (pyannote)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from .. import device as dev
from ..audio_util import read_wav, write_wav
from ..protocol import EngineError

log = logging.getLogger("dub.sep")


class DemucsEngine:
    """htdemucs: 4 fontes (vocals, drums, bass, other). Voz = vocals; fundo = resto."""

    def __init__(self, model: str = "htdemucs", device: str | None = None, **_: object) -> None:
        try:
            import torch  # type: ignore
            from demucs.apply import apply_model  # type: ignore
            from demucs.pretrained import get_model  # type: ignore
        except ImportError as e:
            raise EngineError("demucs não instalado: pip install demucs") from e
        self.device = dev.resolve(device)
        self.model_name = model
        log.info("carregando demucs %s em %s", model, self.device)
        self.model = get_model(model).to(self.device).eval()
        self._apply = apply_model
        self._torch = torch

    def info(self) -> dict:
        return {"name": "demucs", "model": self.model_name, "device": self.device, "capabilities": ["separate"]}

    def separate(self, audio: str, out_dir: str) -> dict:
        torch = self._torch
        data, sr = read_wav(audio)
        model_sr = self.model.samplerate
        wav = torch.from_numpy(data).float()
        if sr != model_sr:
            import torchaudio  # type: ignore

            wav = torchaudio.functional.resample(wav, sr, model_sr)
        # demucs espera (batch, canais, amostras) estéreo
        mix = wav.unsqueeze(0).repeat(2, 1).unsqueeze(0).to(self.device)
        ref_mean, ref_std = mix.mean(), mix.std() + 1e-8
        mix = (mix - ref_mean) / ref_std
        with torch.inference_mode():
            sources = self._apply(self.model, mix, device=self.device, shifts=1, split=True, overlap=0.25, progress=False)[0]
        sources = sources * ref_std + ref_mean
        names = self.model.sources
        vocals = sources[names.index("vocals")].mean(0).cpu().numpy()
        background = sum(sources[i] for i, n in enumerate(names) if n != "vocals").mean(0).cpu().numpy()

        out = Path(out_dir)
        voice_p, bg_p = out / "voice.wav", out / "background.wav"
        # voz volta para 16 kHz (whisper); fundo fica no sample rate do modelo (44.1k)
        voice16 = _resample_np(vocals, model_sr, 16000)
        write_wav(voice_p, voice16, 16000)
        write_wav(bg_p, background, model_sr)
        return {"voice": str(voice_p), "background": str(bg_p)}


def _resample_np(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    n_out = int(round(len(x) * sr_out / sr_in))
    xp = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    xq = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(xq, xp, x).astype(np.float32)


class PyannoteEngine:
    """
    pyannote/speaker-diarization-3.1. Precisa de token Hugging Face
    (variável HF_TOKEN) e de aceitar as condições do modelo no site.
    """

    def __init__(self, model: str = "pyannote/speaker-diarization-3.1", device: str | None = None, **_: object) -> None:
        try:
            import torch  # type: ignore
            from pyannote.audio import Pipeline  # type: ignore
        except ImportError as e:
            raise EngineError("pyannote.audio não instalado: pip install pyannote.audio") from e
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise EngineError("defina HF_TOKEN para usar o pyannote")
        self.device = dev.resolve(device)
        self.model_name = model
        log.info("carregando %s", model)
        self.pipe = Pipeline.from_pretrained(model, use_auth_token=token).to(torch.device(self.device))
        self._speakers: list[str] = []

    def info(self) -> dict:
        return {"name": "pyannote", "model": self.model_name, "device": self.device, "capabilities": ["diarize"]}

    def diarize(self, audio: str, segments: list[dict]) -> dict:
        diar = self.pipe(audio)
        turns = [(t.start, t.end, spk) for t, _, spk in diar.itertracks(yield_label=True)]
        speakers = sorted({s for _, _, s in turns})
        self._speakers = speakers

        for seg in segments:
            s0, s1 = seg["start"], seg["end"]
            overlap: dict[str, float] = {}
            for t0, t1, spk in turns:
                ov = min(s1, t1) - max(s0, t0)
                if ov > 0:
                    overlap[spk] = overlap.get(spk, 0.0) + ov
            if overlap:
                dominant = max(overlap, key=overlap.get)  # type: ignore[arg-type]
                seg["speaker"] = dominant
                # sobreposição: outro falante cobre >30% do trecho
                dur = max(s1 - s0, 1e-6)
                seg["overlapped"] = any(v / dur > 0.3 for k, v in overlap.items() if k != dominant)
            else:
                seg["speaker"] = speakers[0] if speakers else "SPEAKER_00"
                seg["overlapped"] = False
        return {"segments": segments, "speakers": speakers}
