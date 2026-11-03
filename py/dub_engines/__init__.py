"""
dub_engines — engines de IA em Python expostas ao core em Rust.

Cada engine é um processo separado que fala JSON-lines pelo stdin/stdout
(ver `protocol.py`). O Rust nunca importa Python: só sobe o processo com
`python -m dub_engines <engine> [opções]` e troca mensagens.

Engines disponíveis (ver `registry.py`):
  whisper   — transcrição (faster-whisper)
  marian    — tradução (Helsinki-NLP opus-mt, via transformers)
  nllb      — tradução (NLLB-200, via transformers)
  piper     — TTS leve, vozes PT-BR prontas
  kokoro    — TTS natural, leve
  xtts      — TTS com clonagem de voz (Coqui XTTS v2)
  demucs    — separação voz/fundo
  pyannote  — diarização
  mock      — engine falsa para testes (sem dependências)
"""

__version__ = "0.1.0"
