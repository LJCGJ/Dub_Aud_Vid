# Dub_Aud_Vid

Dublagem automática de vídeos com IA, 100% local. Recebe um vídeo (arquivo ou
URL do YouTube) em inglês e devolve o mesmo vídeo com uma faixa de áudio em
PT-BR — voz dublada por cima da trilha original, áudio original mantido como
segunda faixa e legenda embutida, como no YouTube.

```
dub video "https://www.youtube.com/watch?v=..."              # whisper.cpp + Piper nativos, voz pt_BR-faber-medium
dub video aula.mp4 --whisper-model large-v3 --voice pt_BR-edresson-low
dub video aula.mp4 --tts kokoro --voice pf_dora                 # TTS via Python
dub doctor
```

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
| `dub-native` | Rust + C++ | Engines nativas: WSOLA (C++ via `cxx`), whisper.cpp (`whisper-rs`), Piper via ONNX Runtime + espeak-ng |
| `dub-cli` | Rust | Comando `dub`; tenta engine nativa → cai para Python |
| `py/dub_engines` | Python | Engines: faster-whisper, Marian/NLLB, Piper, Kokoro, XTTS, Demucs, pyannote |
| `native/timestretch` | C++ | Time-stretch WSOLA (preserva o tom), usado pelo `dub-native` |

O core em Rust **não depende de Python** para compilar ou rodar. Toda engine
fica atrás dos traits `Transcriber`, `Translator`, `Synthesizer`, `Separator` e
`Diarizer`. A CLI resolve cada papel assim:

| Papel | Nativa (Rust/C++) | Python (fallback) |
|---|---|---|
| Transcrição | whisper.cpp (modelo ggml) | faster-whisper |
| Tradução | — (Parte 2b: CTranslate2) | Marian / NLLB |
| Síntese | Piper via ONNX Runtime | Piper / Kokoro / XTTS |
| Separação voz/fundo | — (Parte 2b: demucs.cpp) | Demucs |
| Diarização | — (Parte 2b: sherpa-onnx) | pyannote |
| Time-stretch | WSOLA em C++ | ffmpeg `atempo` (`--ffmpeg-stretch`) |

`--transcriber native|python|auto` e `--tts-impl native|python|auto` forçam um lado.

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

As engines nativas procuram arquivos em `DUB_MODELS_DIR/`, `~/.cache/dub/` e
ao lado do executável:

```
~/.cache/dub/
├── whisper/ggml-medium.bin            # https://huggingface.co/ggerganov/whisper.cpp
└── voices/pt_BR-faber-medium.onnx     # https://huggingface.co/rhasspy/piper-voices
           pt_BR-faber-medium.onnx.json
```

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
  --translator marian      marian | nllb
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
cargo clippy --workspace
dub video teste.mp4 --mock    # roda o pipeline inteiro com engines falsas, sem modelos
```

Para adicionar uma engine Python: crie a classe em `py/dub_engines/engines/`,
registre em `registry.py` e implemente os métodos do protocolo
(`py/dub_engines/protocol.py`). Nenhuma mudança em Rust é necessária.

## Roadmap

- [x] **Parte 1** — core, pipeline paralelo, engines Python, CLI, adaptador vídeo/YouTube
- [x] **Parte 2** — WSOLA em C++, whisper.cpp nativo, Piper nativo (ONNX Runtime + espeak-ng), fallback nativa → Python
- [ ] **Parte 2b** — tradução nativa (CTranslate2), separação (demucs.cpp), diarização (sherpa-onnx); build 100% sem Python
- [ ] **Parte 3** — UI Tauri, gerenciador/download de pacotes de voz, voz por falante, adaptador GameDub, instalador Windows

## Licença

MIT. Modelos de terceiros têm licenças próprias (XTTS v2: CPML, não comercial).
