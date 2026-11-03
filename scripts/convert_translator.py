#!/usr/bin/env python3
"""
Converte um modelo Marian/OPUS-MT do Hugging Face para CTranslate2 (int8) no
layout que o `dub` espera:

    <saida>/<nome>/
        model.bin, config.json, shared_vocabulary.json   (CTranslate2)
        source.spm, target.spm                           (SentencePiece)
        dub.json                                         (metadados: par de idiomas, prefixo)

Uso:
    python scripts/convert_translator.py Helsinki-NLP/opus-mt-tc-big-en-pt \
        --name opus-mt-tc-big-en-pt-ct2-int8 --target-prefix ">>por_BR<<" --out dist/

Depois: tar -czf <nome>.tar.gz -C dist <nome>
Requer: pip install ctranslate2 transformers sentencepiece torch
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_model", help="id no Hugging Face, ex.: Helsinki-NLP/opus-mt-tc-big-en-pt")
    ap.add_argument("--name", required=True, help="nome da pasta de saída (id usado em --translator-model)")
    ap.add_argument("--out", default="dist", help="pasta onde criar <nome>/")
    ap.add_argument("--quantization", default="int8", choices=["int8", "int8_float16", "float16", "float32"])
    ap.add_argument("--target-prefix", default=None, help='token de variante, ex. ">>por_BR<<" (modelos multi-alvo)')
    ap.add_argument("--source-lang", default="en")
    ap.add_argument("--target-lang", default="pt-BR")
    ap.add_argument("--tar", action="store_true", help="também gera <nome>.tar.gz em --out")
    args = ap.parse_args()

    out_dir = Path(args.out) / args.name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] convertendo {args.hf_model} → {out_dir} ({args.quantization})")
    subprocess.run(
        [
            sys.executable, "-m", "ctranslate2.converters.transformers",
            "--model", args.hf_model,
            "--output_dir", str(out_dir),
            "--quantization", args.quantization,
            "--copy_files", "source.spm", "target.spm",
        ],
        check=True,
    )

    print("[2/3] verificando arquivos")
    for f in ("model.bin", "config.json", "source.spm", "target.spm"):
        if not (out_dir / f).exists():
            # Alguns modelos usam um único vocab (`spiece.model`); copiamos como ambos.
            from huggingface_hub import hf_hub_download  # type: ignore

            if f in ("source.spm", "target.spm"):
                src = hf_hub_download(args.hf_model, "spiece.model")
                shutil.copy(src, out_dir / f)
            else:
                raise SystemExit(f"arquivo esperado não gerado: {f}")

    meta = {
        "hf_model": args.hf_model,
        "quantization": args.quantization,
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "target_prefix": args.target_prefix,
        "format": "ctranslate2",
        "dub_layout": 1,
    }
    (out_dir / "dub.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    size_mb = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()) / 1e6
    print(f"[3/3] pronto: {out_dir} ({size_mb:.0f} MB)")

    if args.tar:
        tar_path = Path(args.out) / f"{args.name}.tar.gz"
        subprocess.run(["tar", "-czf", str(tar_path), "-C", str(Path(args.out)), args.name], check=True)
        print(f"      {tar_path} ({tar_path.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
