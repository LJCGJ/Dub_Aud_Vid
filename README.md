# Dub_Aud_Vid

[![CI](https://github.com/LJCGJ/Dub_Aud_Vid/actions/workflows/ci.yml/badge.svg)](https://github.com/LJCGJ/Dub_Aud_Vid/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/LJCGJ/Dub_Aud_Vid?include_prereleases)](https://github.com/LJCGJ/Dub_Aud_Vid/releases)

Dublagem automática de vídeos com IA, 100% local. Recebe um vídeo (arquivo ou
URL do YouTube) em inglês e devolve o mesmo vídeo com uma faixa de áudio em
PT-BR — voz dublada por cima da trilha original, áudio original mantido como
segunda faixa e legenda embutida, como no YouTube.

```
dub models download                                 # whisper medium + voz pt_BR-faber + tradutor en→pt (~1,8 GB)
dub video "https://www.youtube.com/watch?v=..."     # 100% nativo: whisper.cpp → CTranslate2 → Piper → remux
dub video aula.mp4 --whisper-model large-v3-turbo-q5_0 --voice pt_BR-edresson-low
dub video aula.mp4 --tts kokoro --voice pf_dora     # TTS via Python (opcional)
dub doctor
```

**Começo rápido (Windows):** baixe o zip da [última release](https://github.com/LJCGJ/Dub_Aud_Vid/releases),
extraia, e rode `dub.exe models download` seguido de `dub.exe video <arquivo ou URL>`.
Não precisa de Python para o fluxo principal.

## Como funciona

```
 vídeo ─► áudio 16k ─► [demucs] ─► voz ─► [whisper] ─► [pyannote] ─► segmentos
                          │                                            │
                        fundo                             tradução (CPU) ║ síntese (GPU)
                          │                                            │
                          └─────────────► time-fit + mixagem ◄─────────┘
                                                   │
                                        remux: vídeo + dub + original + SRT
```

- **Tradução e síntese rodam em paralelo**: a CPU traduz o lote N enquanto a
  GPU sintetiza o N-1. Sem GPU, tudo roda em CPU automaticamente.
- **Time-fit**: português costuma ficar 15–30% mais longo que inglês. Cada fala
  é acelerada só o necessário (máx. 1.35x por padrão, preservando o tom via
  ffmpeg `atempo`), usando o silêncio seguinte como folga antes de acelerar.
- **Sobreposição de falas**: trechos onde dois falantes se cruzam têm a
  dublagem com volume reduzido para não brigar com a trilha.

## Arquitetura

| Crate | Linguagem | Papel |
|---|---|---|
| `dub-core` | Rust | Tipos, traits das engines, `Backend` (GPU/CPU), pipeline paralelo, time-fit |
| `dub-media` | Rust | ffmpeg/ffprobe/yt-dlp: extração, atempo, remux |
| `dub-engines-py` | Rust | Sobe engines Python como processos e fala JSON-lines com elas |
| `dub-input-video` | Rust | Adaptador vídeo/YouTube (`prepare` → pipeline → `finish`) |
| `dub-native` | Rust + C++ | Engines nativas: WSOLA (C++ via `cxx`), whisper.cpp, CTranslate2, Piper via ONNX Runtime + espeak-ng |
| `dub-models` | Rust | Catálogo e download de modelos (`dub models`) |
| `dub-cli` | Rust | Comando `dub`; tenta engine nativa → cai para Python |
| `py/dub_engines` | Python | Engines: faster-whisper, Marian/NLLB, Piper, Kokoro, XTTS, Demucs, pyannote |
| `native/timestretch` | C++ | Time-stretch WSOLA (preserva o tom), usado pelo `dub-native` |

O core em Rust **não depende de Python** para compilar ou rodar. Toda engine
fica atrás dos traits `Transcriber`, `Translator`, `Synthesizer`, `Separator` e
`Diarizer`. A CLI resolve cada papel assim:

| Papel | Nativa (Rust/C++) | Python (fallback) |
|---|---|---|
| Transcrição | whisper.cpp (modelo ggml) | faster-whisper |
| Tradução | CTranslate2 (Marian int8) | Marian / NLLB (transformers) |
| Síntese | Piper via ONNX Runtime | Piper / Kokoro / XTTS |
| Separação voz/fundo | — (Parte 2b: demucs.cpp) | Demucs |
| Diarização | — (Parte 2b: sherpa-onnx) | pyannote |
| Time-stretch | WSOLA em C++ | ffmpeg `atempo` (`--ffmpeg-stretch`) |

`--transcriber`, `--translator` e `--tts-impl` aceitam `native | python | auto` para forçar um lado.

## Instalação

Requisitos: Rust 1.75+, **cmake + compilador C++** (Visual Studio Build Tools no
Windows) para as engines nativas, Python 3.10+, ffmpeg no PATH.

> **Máquina com Smart App Control / WDAC** (comum em PC corporativo): o cargo
> não consegue executar os build scripts que compila (`Acesso negado (os error 5)`)
> e nenhum `.exe` não assinado roda. Não há contorno local — compile em um PC
> sem essa política.

```bash
# 1. Build (escolha o backend de GPU da máquina de build)
cargo build --release                                   # CPU
cargo build --release --features native-cuda            # NVIDIA (CUDA Toolkit instalado)
cargo build --release --features native-vulkan          # AMD / Intel / NVIDIA via Vulkan SDK
cargo build --release --features native-directml        # Piper em GPU no Windows (qualquer fabricante)
cargo build --release --no-default-features             # sem engines nativas (só Python)

# 2. Ambiente Python das engines
cd py
python -m venv .venv
.venv/bin/pip install -e ".[basic,kokoro,youtube]"      # Linux/macOS
# .venv\Scripts\pip install -e ".[basic,kokoro,youtube]"  # Windows

# GPU NVIDIA: instale o torch com CUDA ANTES das engines
# .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124

# 3. Confere o que foi detectado
./target/release/dub doctor
```

Extras opcionais:

| Extra | Instala | Para quê |
|---|---|---|
| `separate` | demucs | separar voz do fundo (recomendado; GPU ajuda muito) |
| `diarize` | pyannote.audio | identificar falantes — exige `HF_TOKEN` e aceitar os termos do modelo no Hugging Face |
| `clone` | coqui-tts (XTTS v2) | clonar a voz do falante original (GPU recomendada; licença não comercial) |

## Modelos e pacotes de voz

`dub models list` mostra o catálogo; `dub models download [nomes]` baixa para a
primeira pasta gravável entre `DUB_MODELS_DIR`, `<exe>/models` (layout portátil
do zip) e `~/.cache/dub` (`%LOCALAPPDATA%\dub` no Windows):

```
models/
├── whisper/ggml-medium.bin                       # huggingface.co/ggerganov/whisper.cpp
├── voices/pt_BR-faber-medium.onnx(.json)         # huggingface.co/rhasspy/piper-voices
└── translate/opus-mt-tc-big-en-pt-ct2-int8/      # release models-v1 deste repositório
```

Os tradutores são modelos Marian do Helsinki-NLP convertidos para CTranslate2
int8 pelo workflow `models.yml` (`scripts/convert_translator.py`) e publicados
na release `models-v1`. Para adicionar um par de idiomas, inclua-o na matriz do
workflow e no catálogo (`crates/dub-models/src/catalog.rs`).

- **Piper** (padrão, nativo): `--voice pt_BR-faber-medium` (masc.) ou
  `pt_BR-edresson-low`. Um pacote de voz = `.onnx` + `.onnx.json`.
  Precisa da `onnxruntime` (DLL ao lado do exe, `DUB_ORT_LIB`, ou a do pacote
  Python `onnxruntime`) e da pasta `espeak-ng-data` (ao lado do exe,
  `DUB_ESPEAK_DATA`, ou a do pacote Python `piper-tts`). O `dub doctor` mostra
  o que foi encontrado.
- **Kokoro** (Python): `--tts kokoro --voice pf_dora` (fem.) / `pm_alex`, `pm_santa`.
- **XTTS** (Python, clonagem): `--tts xtts`. Usa o próprio trecho do falante como referência.

## Uso

```
dub video <arquivo|url> [opções]

  --out <arquivo>          saída (padrão: <título>.dub.pt-BR.mp4)
  --from en --to pt-BR     idiomas
  --cpu                    força CPU
  --transcriber auto       auto | native | python
  --whisper-model medium   tiny | base | small | medium | large-v3 | caminho ggml-*.bin
  --translator auto        auto | native | python
  --translator-model <x>   nativo: pasta CT2 (padrão opus-mt-tc-big-en-pt-ct2-int8); python: marian | nllb
  --tts piper              piper | kokoro | xtts
  --tts-impl auto          auto | native | python (só piper)
  --voice <nome|.onnx>     padrão: pt_BR-faber-medium
  --ffmpeg-stretch         atempo do ffmpeg em vez do WSOLA nativo
  --no-separate            desliga demucs
  --no-diarize             desliga pyannote
  --no-original-track      não manter o áudio original
  --no-subtitles           não embutir SRT
  --background-gain 0.9    volume do fundo
  --max-speed 1.35         aceleração máxima da fala
  -v / -vv                 logs
```

Arquivos intermediários ficam em `dub_work/<título>/` (áudio extraído, voz e
fundo separados, cada segmento sintetizado, `dub.srt`, `dub_mix.wav`).

## Desenvolvimento

```bash
cargo test --workspace        # inclui teste ponta a ponta com engines mock em Python
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
dub video teste.mp4 --mock    # roda o pipeline inteiro com engines falsas, sem modelos
```

CI (`.github/workflows/ci.yml`) roda testes, clippy e um smoke test da CLI em
Linux, Windows e macOS a cada push. `git tag v0.1.0 && git push --tags` dispara
`release.yml`, que publica zips por plataforma com `dub`, `onnxruntime` e
`espeak-ng-data`.

Para adicionar uma engine Python: crie a classe em `py/dub_engines/engines/`,
registre em `registry.py` e implemente os métodos do protocolo
(`py/dub_engines/protocol.py`). Nenhuma mudança em Rust é necessária.

## Roadmap

- [x] **Parte 1** — core, pipeline paralelo, engines Python, CLI, adaptador vídeo/YouTube
- [x] **Parte 2** — WSOLA em C++, whisper.cpp nativo, Piper nativo (ONNX Runtime + espeak-ng), fallback nativa → Python
- [x] **Parte 2b** — tradução nativa (CTranslate2), `dub models`, CI em 3 SOs, releases por tag, workflow de conversão de modelos. Fluxo principal 100% sem Python
- [ ] **Parte 3** — UI Tauri, voz por falante, adaptador GameDub, instalador Windows
- [ ] **Depois** — separação (demucs.cpp) e diarização (sherpa-onnx) nativas

## Licença

MIT. Modelos de terceiros têm licenças próprias (XTTS v2: CPML, não comercial).
