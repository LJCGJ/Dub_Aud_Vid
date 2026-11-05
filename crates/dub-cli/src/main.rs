mod models_cmd;
mod wizard;

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
    /// Sem subcomando, abre o assistente interativo (menus com as setas).
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Dubla um vídeo local ou URL do YouTube (sem argumentos: assistente interativo)
    Video(Box<VideoArgs>),
    /// Mostra o hardware detectado e as ferramentas encontradas
    Doctor,
    /// Lista e baixa modelos (whisper, vozes, tradutores)
    Models(models_cmd::ModelsArgs),
}

#[derive(Args)]
struct VideoArgs {
    /// Caminho do vídeo ou URL (omita para o assistente perguntar)
    source: Option<String>,
    /// Abre o assistente interativo mesmo com argumentos (as respostas sobrescrevem as flags)
    #[arg(short, long)]
    interactive: bool,
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
    /// Força CPU mesmo com GPU disponível (atalho para --device cpu)
    #[arg(long)]
    cpu: bool,
    /// Onde rodar: auto (decide pelo hardware) | cpu | gpu | split (whisper na CPU, voz na GPU)
    #[arg(long, default_value = "auto")]
    device: dub_core::DeviceMode,
    /// Uso do hardware: eco (metade dos núcleos, sem GPU) | balanced (deixa folga; padrão) | max
    #[arg(long, default_value = "balanced")]
    performance: dub_core::Performance,

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
    /// Desliga a separação voz/instrumental (a música fica só abaixada sob a fala)
    #[arg(long)]
    no_separate: bool,
    /// Modelo de separação: Kim_Vocal_2 (padrão), UVR-MDX-NET-Inst_HQ_3, caminho .onnx ou "demucs" (Python)
    #[arg(long)]
    separator_model: Option<String>,
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
    // Console do Windows em UTF-8, senão acentos saem como "Ã£".
    #[cfg(windows)]
    unsafe {
        windows_sys::Win32::System::Console::SetConsoleOutputCP(65001);
    }
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
        None => {
            // Argumentos padrão de `dub video`, preenchidos pelo assistente.
            #[derive(Parser)]
            struct Only {
                #[command(flatten)]
                v: VideoArgs,
            }
            let mut a = Only::parse_from(["dub"]).v;
            apply_wizard(&mut a, None)?;
            video(a)
        }
        Some(Cmd::Doctor) => doctor(),
        Some(Cmd::Models(args)) => models_cmd::run(args),
        Some(Cmd::Video(args)) => {
            let mut a = *args;
            if a.source.is_none() || a.interactive {
                let initial = a.source.clone();
                apply_wizard(&mut a, initial)?;
            }
            video(a)
        }
    }
}

/// Preenche os argumentos com as respostas do assistente.
fn apply_wizard(a: &mut VideoArgs, initial: Option<String>) -> Result<()> {
    let c = wizard::run(initial)?;
    a.source = Some(c.source);
    a.device = c.device;
    a.performance = c.performance;
    a.whisper_model = c.whisper_model;
    a.voice = Some(c.voice);
    if a.out.is_none() {
        a.out = c.out;
    }
    Ok(())
}

fn doctor() -> Result<()> {
    let r = dub_runtime::doctor();
    println!("Hardware");
    println!("  dispositivo preferido: {}", r.device);
    println!("  disponíveis:           {}", r.devices.join(", "));
    println!("  threads de CPU:        {}", r.cpu_threads);
    if let Some(g) = &r.gpu_name {
        println!(
            "  GPU:                   {g}{}{}",
            r.gpu_vram_mb.map(|v| format!(" ({v} MB VRAM)")).unwrap_or_default(),
            if r.gpu_integrated { " — integrada, divide a RAM com o sistema" } else { "" }
        );
    }
    println!(
        "  backends de GPU:       {}",
        if r.gpu_backends.is_empty() { "nenhum neste build (tudo em CPU)".to_string() } else { r.gpu_backends.join(", ") }
    );
    if let (Some(t), Some(f)) = (r.ram_total_mb, r.ram_free_mb) {
        println!("  RAM:                   {:.1} GB livres de {:.1} GB", f as f64 / 1024.0, t as f64 / 1024.0);
    }
    println!("  plano automático:");
    for p in &r.plan {
        println!("    {:<22} → {:<9} {}", p.role.label(), p.device.to_string(), p.reason);
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
        source: a.source.expect("fonte definida pelo assistente ou pela linha de comando"),
        output: a.out,
        output_dir: None,
        work_dir: a.work_dir,
        source_lang: if a.from.is_empty() { None } else { Some(a.from) },
        target_lang: a.to,
        force_cpu: a.cpu,
        performance: a.performance,
        device_mode: a.device,
        engines: engines::EngineChoice {
            transcriber_impl: a.transcriber,
            whisper_model: a.whisper_model,
            translator_impl: a.translator,
            translator_model: a.translator_model,
            tts: a.tts,
            tts_impl: a.tts_impl,
            voice: a.voice,
            separate: !a.no_separate,
            separator_model: a.separator_model,
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
        JobEvent::Device { gpu_name, mode, plan, ram_free_mb, .. } => {
            eprintln!(
                "▶ hardware: {}{}  |  modo: {mode}",
                gpu_name.as_deref().unwrap_or("sem GPU detectada"),
                ram_free_mb.map(|m| format!(", {:.1} GB de RAM livre", m as f64 / 1024.0)).unwrap_or_default()
            );
            for p in plan {
                eprintln!("  {:<22} → {:<9} {}", p.role.label(), p.device.to_string(), p.reason);
            }
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
    eprintln!("  legenda:  {} (tradução) e {} (original reconhecido)", r.srt.display(), r.srt_source.display());
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
