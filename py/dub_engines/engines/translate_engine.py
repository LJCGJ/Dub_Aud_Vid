"""
Tradução offline via transformers.

  marian — Helsinki-NLP/opus-mt-tc-big-en-pt: leve (~230 MB), rápido em CPU,
           qualidade boa para en→pt. Padrão.
  nllb   — facebook/nllb-200-distilled-600M: 200 idiomas, melhor para pares
           raros; mais pesado.
"""

from __future__ import annotations

import logging

from .. import device as dev
from ..protocol import EngineError

log = logging.getLogger("dub.translate")

MARIAN_MODELS = {
    ("en", "pt"): "Helsinki-NLP/opus-mt-tc-big-en-pt",
    ("pt", "en"): "Helsinki-NLP/opus-mt-tc-big-pt-en",
    ("es", "pt"): "Helsinki-NLP/opus-mt-es-pt",
    ("ja", "pt"): "Helsinki-NLP/opus-mt-ja-pt",
}

NLLB_CODES = {"en": "eng_Latn", "pt": "por_Latn", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn", "ja": "jpn_Jpan", "ko": "kor_Hang", "zh": "zho_Hans", "it": "ita_Latn", "ru": "rus_Cyrl"}


def _base(lang: str) -> str:
    return lang.split("-")[0].split("_")[0].lower()


class _Base:
    def __init__(self, device: str | None = None, threads: int | None = None) -> None:
        try:
            import torch  # type: ignore  # noqa: F401
            import transformers  # type: ignore  # noqa: F401
        except ImportError as e:
            raise EngineError("transformers/torch não instalados: pip install torch transformers sentencepiece") from e
        self.device = dev.resolve(device)
        dev.cpu_threads(threads)
        self._models: dict = {}

    def _generate(self, tok, model, texts: list[str], **gen) -> list[str]:
        import torch  # type: ignore

        out: list[str] = []
        bs = 8
        for i in range(0, len(texts), bs):
            chunk = texts[i : i + bs]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.inference_mode():
                gen_ids = model.generate(**enc, num_beams=4, max_new_tokens=512, **gen)
            out.extend(t.strip() for t in tok.batch_decode(gen_ids, skip_special_tokens=True))
        return out


class MarianEngine(_Base):
    def __init__(self, model: str | None = None, **kw: object) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self.override = model

    def info(self) -> dict:
        return {"name": "marian", "model": self.override or "opus-mt (auto por par)", "device": self.device, "capabilities": ["translate"]}

    def _load(self, src: str, tgt: str):
        key = (src, tgt)
        if key not in self._models:
            from transformers import MarianMTModel, MarianTokenizer  # type: ignore

            name = self.override or MARIAN_MODELS.get(key)
            if not name:
                raise EngineError(f"sem modelo Marian para {src}→{tgt}; use a engine nllb")
            log.info("carregando %s", name)
            tok = MarianTokenizer.from_pretrained(name)
            model = MarianMTModel.from_pretrained(name).to(self.device).eval()
            self._models[key] = (tok, model)
        return self._models[key]

    def translate(self, texts: list[str], **kw: str) -> dict:
        src, tgt = _base(kw.get("from", "en")), _base(kw.get("to", "pt"))
        tok, model = self._load(src, tgt)
        # Modelos tc-big multi-alvo aceitam prefixo de variante; pt-BR quando pedido.
        prefix = ">>por_BR<< " if kw.get("to", "").lower() in ("pt-br", "pt_br") and "tc-big" in (self.override or MARIAN_MODELS.get((src, tgt), "")) else ""
        inputs = [prefix + t if t.strip() else t for t in texts]
        res = self._generate(tok, model, inputs)
        return {"texts": [r if t.strip() else "" for r, t in zip(res, texts)]}


class NllbEngine(_Base):
    def __init__(self, model: str = "facebook/nllb-200-distilled-600M", **kw: object) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

        self.model_name = model
        log.info("carregando %s", model)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model).to(self.device).eval()

    def info(self) -> dict:
        return {"name": "nllb", "model": self.model_name, "device": self.device, "capabilities": ["translate"]}

    def translate(self, texts: list[str], **kw: str) -> dict:
        src, tgt = _base(kw.get("from", "en")), _base(kw.get("to", "pt"))
        try:
            self.tok.src_lang = NLLB_CODES[src]
            tgt_id = self.tok.convert_tokens_to_ids(NLLB_CODES[tgt])
        except KeyError as e:
            raise EngineError(f"idioma sem código NLLB mapeado: {e}") from e
        res = self._generate(self.tok, self.model, texts, forced_bos_token_id=tgt_id)
        return {"texts": res}
