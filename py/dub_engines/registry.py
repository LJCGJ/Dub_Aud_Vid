"""Mapa nome → classe da engine. Importa de forma preguiçosa para que uma
dependência faltando em uma engine não derrube as outras."""

from __future__ import annotations

from importlib import import_module
from typing import Any

ENGINES: dict[str, tuple[str, str]] = {
    "mock": ("dub_engines.engines.mock", "MockEngine"),
    "whisper": ("dub_engines.engines.whisper_engine", "WhisperEngine"),
    "marian": ("dub_engines.engines.translate_engine", "MarianEngine"),
    "nllb": ("dub_engines.engines.translate_engine", "NllbEngine"),
    "piper": ("dub_engines.engines.tts_engines", "PiperEngine"),
    "kokoro": ("dub_engines.engines.tts_engines", "KokoroEngine"),
    "xtts": ("dub_engines.engines.tts_engines", "XttsEngine"),
    "demucs": ("dub_engines.engines.separate_diarize", "DemucsEngine"),
    "pyannote": ("dub_engines.engines.separate_diarize", "PyannoteEngine"),
}


def create(name: str, **kwargs: Any) -> Any:
    try:
        mod_name, cls_name = ENGINES[name]
    except KeyError as e:
        raise KeyError(f"engine desconhecida '{name}'. Disponíveis: {', '.join(ENGINES)}") from e
    cls = getattr(import_module(mod_name), cls_name)
    return cls(**kwargs)
