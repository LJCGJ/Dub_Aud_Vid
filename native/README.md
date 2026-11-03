# native/

Código C++ próprio do projeto (Parte 2), exposto ao Rust via `cxx`:

- `timestretch/` — WSOLA/phase-vocoder para o time-fit sem depender do ffmpeg
- `glue/` — adaptadores para whisper.cpp, CTranslate2, sherpa-onnx e demucs.cpp
  onde os bindings Rust existentes não cobrirem o necessário

Vazio na Parte 1 de propósito: todas as engines rodam via Python.
