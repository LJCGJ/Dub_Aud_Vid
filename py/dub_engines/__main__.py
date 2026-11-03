"""
Ponto de entrada: `python -m dub_engines <engine> [--model X] [--device cuda|cpu] [--threads N] [--opt chave=valor ...]`

Sobe a engine, anuncia `ready` e entra no loop do protocolo.
"""

from __future__ import annotations

import argparse
import logging
import sys

from . import protocol, registry


def main() -> int:
    ap = argparse.ArgumentParser(prog="dub_engines")
    ap.add_argument("engine", choices=sorted(registry.ENGINES))
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None, help="cuda | cpu | mps | auto")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--opt", action="append", default=[], help="opção extra chave=valor (repetível)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    # stdout é do protocolo; logs vão para stderr.
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[py:%(name)s] %(levelname)s %(message)s",
    )
    # Bibliotecas barulhentas
    for noisy in ("transformers", "torch", "numba", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    kwargs: dict = {"device": args.device, "threads": args.threads}
    if args.model:
        kwargs["model"] = args.model
    for opt in args.opt:
        if "=" not in opt:
            ap.error(f"--opt espera chave=valor, recebeu '{opt}'")
        k, v = opt.split("=", 1)
        kwargs[k] = v

    try:
        engine = registry.create(args.engine, **kwargs)
        info = engine.info()
    except Exception as e:  # noqa: BLE001
        logging.getLogger("dub").exception("falha ao iniciar engine")
        protocol.send_not_ready(f"{type(e).__name__}: {e}")
        return 2

    protocol.send_ready(info)
    protocol.serve(engine)
    return 0


if __name__ == "__main__":
    sys.exit(main())
