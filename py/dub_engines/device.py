"""
Resolução de dispositivo: o Rust manda uma *preferência* (cuda/cpu/mps); aqui
confirmamos se ela realmente funciona e caímos para CPU se não.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("dub.device")


def resolve(preferred: str | None) -> str:
    """Devolve 'cuda', 'mps' ou 'cpu' — sempre algo que funciona nesta máquina."""
    pref = (preferred or "auto").lower()
    try:
        import torch  # type: ignore
    except ImportError:
        if pref not in ("cpu", "auto"):
            log.warning("torch não instalado; usando CPU")
        return "cpu"

    if pref in ("cuda", "auto") and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        log.info("usando CUDA: %s", name)
        return "cuda"
    if pref in ("mps", "auto") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        log.info("usando Metal (MPS)")
        return "mps"
    if pref not in ("cpu", "auto"):
        log.warning("dispositivo '%s' indisponível; usando CPU", pref)
    return "cpu"


def cpu_threads(requested: int | None) -> int:
    n = requested or os.cpu_count() or 4
    try:
        import torch  # type: ignore

        torch.set_num_threads(n)
    except Exception:  # noqa: BLE001
        pass
    return n


def ct2_compute_type(device: str) -> str:
    """Tipo numérico para CTranslate2/faster-whisper."""
    return "float16" if device == "cuda" else "int8"
