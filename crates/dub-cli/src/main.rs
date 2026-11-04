mod models_cmd;

use dub_runtime::{engines, JobEvent, JobRequest};

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use dub_core::{Progress, Stage};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "dub", version, about = "Dublagem automática de vídeos com IA (EN → PT-BR)")]
struct Cli {
    /// Mais logs (-v info, -vv debug)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Dubla um vídeo local ou URL do YouTube
    Video(Box<VideoArgs>),
    /// Mostra o hardware detectado e as ferramentas encontradas
    Doctor,
    /// Lista e baixa modelos (whisper, vozes, tradutores)
    Models(models_cmd::ModelsArgs),
}

#[derive(Args)]
struct VideoArgs {
    /// Caminho do vídeo ou URL
    source: String,
    /// Arquivo de saída (padrão: <título>.dub.pt-BR.mp4)
    #[arg(short, long)]
    out: Option<PathBuf>,
    /// Idioma de origem (vazio = detectar)
    #[arg(long, default_value = "en")]
    from: String,
    /// Idioma de destino
    #[arg(long, default_value = "pt-BR")]
    to: String,
    /// Diretório de trabalho (padrão: ./dub_work/<título>)
    #[arg(long)]
    work_dir: Option<PathBuf>,
    /// Força CPU mesmo com GPU disponível
    #[arg(long)]
    cpu: bool,

    /// Implementação da transcrição: auto | native (whisper.cpp) | python (faster-whisper)
    #[arg(long, default_value = "auto")]
    transcriber: engines::Impl,
    /// Modelo whisper: tiny, base, small, medium, large-v3 (nome ou caminho do ggml-*.bin)
    #[arg(long, default_value = "medium")]
    whisper_model: String,
    /// Implementação da tradução: auto | native (CTranslate2) | python (marian/nllb via transformers)
    #[arg(long, default_value = "auto")]
    translator: engines::Impl,
    /// Modelo de tradução: nativo → pasta CT2 (padrão opus-mt-tc-big-en-pt-ct2-int8); python → marian | nllb
    #[arg(long)]
    translator_model: Option<String>,
    /// Engine de voz: piper | kokoro | xtts
    #[arg(long, default_value = "piper")]
    tts: String,
    /// Implementação do TTS (só para piper): auto | native (ONNX Runtime) | python
    #[arg(long, default_value = "auto")]
    tts_impl: engines::Impl,
    /// Voz/pacote de voz: piper → nome (pt_BR-faber-medium) ou caminho .onnx; kokoro → pf_dora, pm_alex
    #[arg(long, default_value = "pt_BR-faber-medium")]
    voice: Option<String>,
    /// Usa o ffmpeg (atempo) em vez do WSOLA nativo para ajustar a duração
    #[arg(long)]
    ffmpeg_stretch: bool,
    /// Desliga a separação voz/fundo (demucs)
    #[arg(long)]
    no_separate: bool,
    /// Desliga a identificação de falantes (pyannote)
    #[arg(long)]
    no_diarize: bool,
    /// Não manter o áudio original como segunda faixa
    #[arg(long)]
    no_original_track: bool,
    /// Não embutir legenda
    #[arg(long)]
    no_subtitles: bool,
    /// Volume do fundo original (0–1)
    #[arg(long, default_value_t = 0.9)]
    background_gain: f32,
    /// Velocidade máxima da fala dublada para caber no tempo
    #[arg(long, default_value_t = 1.35)]
    max_speed: f32,
    /// Usa engines falsas (teste do pipeline sem modelos)
    #[arg(long, hide = true)]
    mock: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let level = match cli.verbose {
        0 => "warn",
        1 => "info",
        _ => "debug",
    };
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level)))
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();

    match cli.cmd {
        Cmd::Doctor => doctor(),
        Cmd::Models(args) => models_cmd::run(args),
        Cmd::Video(args) => video(*args),
    }
}

fn doctor() -> Result<()> {
    let r = dub_runtime::doctor();
    println!("Hardware");
    println!("  dispositivo preferido: {}", r.device);
    println!("  disponíveis:           {}", r.devices.join(", "));
    println!("  threads de CPU:        {}", r.cpu_threads);
    if let Some(g) = &r.gpu_name {
        println!("  GPU:                   {g}");
    }
    let print = |t: &dub_runtime::ToolStatus| println!("  {:<26} {} {}", t.name, if t.ok { "ok   " } else { "FALTA" }, t.detail);
    println!("\nFerramentas");
    r.tools.iter().for_each(print);
    println!("\nEngines nativas{}", if r.native_enabled { "" } else { " (desligadas neste build)" });
    r.native.iter().for_each(print);
    println!("\nModelos instalados");
    for kind in ["whisper", "voices", "translate"] {
        let names: Vec<&str> = r.models.iter().filter(|m| m.kind == kind && m.installed).map(|m| m.name.as_str()).collect();
        println!(
            "  {:<10} {}",
            kind,
            if names.is_empty() { "(nenhum) — rode: dub models download".to_string() } else { names.join(", ") }
        );
    }
    println!(
        "\n{}",
        if r.ready_native {
            "✔ pronto para dublar sem Python"
        } else {
            "⚠ ainda falta algo para o fluxo 100% nativo (veja acima)"
        }
    );
    Ok(())
}

fn video(a: VideoArgs) -> Result<()> {
    let req = JobRequest {
        source: a.source,
        output: a.out,
        output_dir: None,
        work_dir: a.work_dir,
        source_lang: if a.from.is_empty() { None } else { Some(a.from) },
        target_lang: a.to,
        force_cpu: a.cpu,
        engines: engines::EngineChoice {
            transcriber_impl: a.transcriber,
            whisper_model: a.whisper_model,
            translator_impl: a.translator,
            translator_model: a.translator_model,
            tts: a.tts,
            tts_impl: a.tts_impl,
            voice: a.voice,
            separate: !a.no_separate,
            diarize: !a.no_diarize,
            mock: a.mock,
        },
        keep_original_audio: !a.no_original_track,
        embed_subtitles: !a.no_subtitles,
        background_gain: a.background_gain,
        max_speed: a.max_speed,
        ffmpeg_stretch: a.ffmpeg_stretch,
    };

    let (tx, rx) = crossbeam_channel::unbounded::<Progress>();
    let ui = std::thread::spawn(move || progress_ui(rx));
    let cancel = Arc::new(AtomicBool::new(false));
    let result = dub_runtime::run_video_job(&req, cancel, |ev| match ev {
        JobEvent::Device { device, gpu_name } => {
            eprintln!("▶ dispositivo: {device}{}", gpu_name.map(|g| format!(" ({g})")).unwrap_or_default())
        }
        JobEvent::Status { message } => eprintln!("▶ {message}…"),
        JobEvent::Opened { title, .. } => eprintln!("▶ vídeo: {title}"),
        JobEvent::Engines { engines } => {
            for e in engines {
                eprintln!(
                    "  {:<12} {} [{}] em {} ({})",
                    e.role,
                    e.name,
                    e.model,
                    e.device,
                    if e.native { "nativa" } else { "python" }
                );
            }
        }
        JobEvent::Progress(p) => {
            let _ = tx.send(p);
        }
        JobEvent::Done(_) | JobEvent::Error { .. } => {}
    });
    drop(ui);
    let r = result?;

    eprintln!("\n✔ pronto em {}m{:02}s", r.elapsed_secs / 60, r.elapsed_secs % 60);
    eprintln!("  vídeo:    {}", r.video.display());
    eprintln!("  legenda:  {}", r.srt.display());
    eprintln!("  segmentos: {}", r.segments.len());
    if r.rushed_segments > 0 {
        eprintln!("  ⚠ {} segmentos acelerados acima de 1.25x (tradução longa)", r.rushed_segments);
    }
    Ok(())
}

/// Barras de progresso por etapa, alimentadas pelo canal do pipeline.
fn progress_ui(rx: crossbeam_channel::Receiver<Progress>) {
    let mp = MultiProgress::new();
    let style = ProgressStyle::with_template("{prefix:<12} [{bar:30}] {pos}/{len} {msg}").unwrap().progress_chars("=> ");
    let mut bars: HashMap<Stage, ProgressBar> = HashMap::new();
    let label = |s: Stage| match s {
        Stage::Separate => "separação",
        Stage::Transcribe => "transcrição",
        Stage::Diarize => "falantes",
        Stage::Translate => "tradução",
        Stage::Synthesize => "síntese",
        Stage::Fit => "ajuste",
        Stage::Mix => "mixagem",
        Stage::Done => "concluído",
    };
    for p in rx.iter() {
        if p.stage == Stage::Done {
            break;
        }
        let bar = bars.entry(p.stage).or_insert_with(|| {
            let b = mp.add(ProgressBar::new(p.total.max(1) as u64));
            b.set_style(style.clone());
            b.set_prefix(label(p.stage));
            b
        });
        bar.set_length(p.total.max(1) as u64);
        bar.set_position(p.done as u64);
        if !p.message.is_empty() {
            bar.set_message(p.message.clone());
        }
        if p.done >= p.total {
            bar.finish();
        }
    }
    for b in bars.values() {
        if !b.is_finished() {
            b.finish();
        }
    }
}
