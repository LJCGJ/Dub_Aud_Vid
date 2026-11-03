# native/

Código C++ próprio do projeto, exposto ao Rust via `cxx` (crate `dub-native`).

- `timestretch/` — WSOLA (Waveform Similarity Overlap-Add): altera a duração
  da fala sem mudar o tom. Usado no time-fit da dublagem. Sem dependências.

Planejado (Parte 2b): adaptadores para CTranslate2, demucs.cpp e sherpa-onnx
onde os bindings Rust não cobrirem o necessário.
