"""
Protocolo JSON-lines entre Rust e as engines Python.

Uma linha JSON por mensagem. stdout é exclusivo do protocolo — qualquer log vai
para stderr (configurado em `server.py`), senão corrompe o canal.

Servidor → cliente, ao iniciar:
    {"ready": true, "info": {"name": "...", "model": "...", "device": "..."}}
  ou, se falhou ao carregar:
    {"ready": false, "error": "..."}

Cliente → servidor:
    {"id": 1, "method": "transcribe", "params": {...}}

Servidor → cliente:
    {"id": 1, "result": {...}}
    {"id": 1, "error": "mensagem"}

Métodos e parâmetros (todos os caminhos são absolutos):
    info        {}                                       → {name, model, device, capabilities: [..]}
    transcribe  {audio, lang?}                           → {segments: [{start, end, text}]}
    translate   {texts: [..], from, to}                  → {texts: [..]}
    synthesize  {text, lang, out_path, speaker_ref?, speed?} → {path, sample_rate, duration}
    separate    {audio, out_dir}                         → {voice, background}
    diarize     {audio, segments: [{index,start,end,text}]} → {segments: [{..., speaker, overlapped}], speakers: [..]}
    shutdown    {}                                       → {}
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Callable


class EngineError(Exception):
    """Erro esperado da engine (vai no campo `error`, sem traceback)."""


def _write(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def send_ready(info: dict[str, Any]) -> None:
    _write({"ready": True, "info": info})


def send_not_ready(error: str) -> None:
    _write({"ready": False, "error": error})


def serve(engine: Any) -> None:
    """
    Loop principal. `engine` é qualquer objeto com métodos nomeados como os do
    protocolo (`transcribe`, `translate`, ...), cada um recebendo **kwargs.
    """
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError as e:
            _write({"id": None, "error": f"JSON inválido: {e}"})
            continue

        msg_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}

        if method == "shutdown":
            _write({"id": msg_id, "result": {}})
            break

        fn: Callable[..., Any] | None = getattr(engine, method, None) if isinstance(method, str) else None
        if fn is None or method.startswith("_"):
            _write({"id": msg_id, "error": f"método desconhecido: {method}"})
            continue

        try:
            result = fn(**params)
            _write({"id": msg_id, "result": result if result is not None else {}})
        except EngineError as e:
            _write({"id": msg_id, "error": str(e)})
        except Exception as e:  # noqa: BLE001 — queremos devolver qualquer falha ao Rust
            traceback.print_exc(file=sys.stderr)
            _write({"id": msg_id, "error": f"{type(e).__name__}: {e}"})
