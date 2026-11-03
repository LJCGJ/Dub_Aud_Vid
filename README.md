# Dub_Aud_Vid

Dublagem automática de vídeos com IA, 100% local. Recebe um vídeo (arquivo ou
URL do YouTube) em inglês e devolve o mesmo vídeo com uma faixa de áudio em
PT-BR — voz dublada por cima da trilha original, áudio original mantido como
segunda faixa e legenda embutida, como no YouTube.

```
dub video "https://www.youtube.com/watch?v=..." --tts kokoro --voice pf_dora
dub video aula.mp4 --tts piper --voice vozes/pt_BR-faber-medium.onnx
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
| `dub-cli` | Rust | Comando `dub` |
| `py/dub_engines` | Python | Engines: faster-whisper, Marian/NLLB, Piper, Kokoro, XTTS, Demucs, pyannote |
| `native/` | C++ | (Parte 2) motores nativos via `cxx`: whisper.cpp, CTranslate2, sherpa-onnx, time-stretch |

O core em Rust **não depende de Python** para compilar ou rodar. Python é um
*plugin de engine* atrás dos traits `Transcriber`, `Translator`, `Synthesizer`,
`Separator` e `Diarizer`; na Parte 2 as implementações nativas em C++ entram
atrás dos mesmos traits, e a CLI passa a tentar nativa → Python.

## Instalação

Requisitos: Rust 1.75+, Python 3.10+, ffmpeg no PATH.

> **Máquina com Smart App Control / WDAC** (comum em PC corporativo): o cargo
> não consegue executar os build scripts que compila (`Acesso negado (os error 5)`)
> e nenhum `.exe` não assinado roda. Não há contorno local — compile em um PC
> sem essa política.

```bash
# 1. Build
cargo build --release

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

## Pacotes de voz

- **Kokoro** (padrão): `--tts kokoro --voice pf_dora` (feminina) ou `pm_alex`,
  `pm_santa` (masculinas). Baixa o modelo na primeira execução.
- **Piper**: baixe um pacote PT-BR (`pt_BR-faber-medium.onnx` + `.json`) de
  https://huggingface.co/rhasspy/piper-voices e passe o `.onnx` em `--voice`.
- **XTTS** (clonagem): `--tts xtts`. Usa o próprio trecho do falante como referência.

## Uso

```
dub video <arquivo|url> [opções]

  --out <arquivo>          saída (padrão: <título>.dub.pt-BR.mp4)
  --from en --to pt-BR     idiomas
  --cpu                    força CPU
  --whisper-model medium   tiny | base | small | medium | large-v3 | distil-large-v3
  --translator marian      marian | nllb
  --tts kokoro             piper | kokoro | xtts
  --voice <nome|.onnx>
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
- [ ] **Parte 2** — engines nativas em C++ (whisper.cpp, CTranslate2, sherpa-onnx, demucs.cpp), time-stretch nativo, fallback nativo → Python, build sem Python
- [ ] **Parte 3** — UI Tauri, gerenciador de pacotes de voz, voz por falante, adaptador GameDub, instalador Windows

## Licença

MIT. Modelos de terceiros têm licenças próprias (XTTS v2: CPML, não comercial).
