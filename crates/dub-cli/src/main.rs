mod engines;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use dub_core::{Backend, Lang, Pipeline, PipelineConfig, Progress, Stage};
use dub_input_video::{VideoInput, VideoOptions};
use dub_media::Ffmpeg;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;
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
    /// Engine de tradução: marian | nllb
    #[arg(long, default_value = "marian")]
    translator: String,
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
        Cmd::Video(args) => video(*args),
    }
}

fn doctor() -> Result<()> {
    let b = Backend::detect();
    println!("Hardware");
    println!("  dispositivo preferido: {}", b.preferred);
    println!("  disponíveis:           {}", b.available.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "));
    println!("  threads de CPU:        {}", b.cpu_threads);
    if let Some(g) = &b.gpu_name {
        println!("  GPU:                   {g}");
    }
    println!("\nFerramentas");
    let check = |name: &str, r: std::result::Result<String, String>| match r {
        Ok(v) => println!("  {name:<10} ok   {v}"),
        Err(e) => println!("  {name:<10} FALTA {e}"),
    };
    check("ffmpeg", Ffmpeg::locate().map(|_| String::new()).map_err(|e| e.to_string()));
    check("yt-dlp", dub_media::YtDlp::locate().map(|_| String::new()).map_err(|e| e.to_string()));
    check(
        "python",
        dub_engines_py::PythonEnv::discover(None).map(|p| p.interpreter.display().to_string()).map_err(|e| e.to_string()),
    );
    doctor_native();
    Ok(())
}

#[cfg(feature = "native")]
fn doctor_native() {
    println!("\nEngines nativas");
    println!("  time-stretch  ok   WSOLA (C++)");
    let found: Vec<String> = dub_native::whisper::model_dirs().into_iter().filter(|d| d.is_dir()).map(|d| d.display().to_string()).collect();
    println!("  whisper.cpp   ok   modelos em: {}", if found.is_empty() { "(nenhuma pasta de modelos ainda)".into() } else { found.join(", ") });
    let ort = dub_native::piper::find_ort_lib();
    let esp = dub_native::phonemize::find_espeak_data();
    match (&ort, &esp) {
        (Some(o), Some(e)) => println!("  piper (ONNX)  ok   onnxruntime: {} | espeak-ng-data: {}", o.display(), e.display()),
        _ => println!(
            "  piper (ONNX)  FALTA {}{}",
            if ort.is_none() { "onnxruntime (DUB_ORT_LIB ou onnxruntime.dll ao lado do exe) " } else { "" },
            if esp.is_none() { "espeak-ng-data (DUB_ESPEAK_DATA)" } else { "" }
        ),
    }
}

#[cfg(not(feature = "native"))]
fn doctor_native() {
    println!("\nEngines nativas: desligadas (compile com a feature 'native')");
}

fn video(a: VideoArgs) -> Result<()> {
    let started = Instant::now();
    let backend = if a.cpu { Backend::cpu_only() } else { Backend::detect() };
    eprintln!("▶ dispositivo: {}{}", backend.preferred, backend.gpu_name.as_deref().map(|g| format!(" ({g})")).unwrap_or_default());

    // 1. Entrada ----------------------------------------------------------
    let tmp_work = a.work_dir.clone().unwrap_or_else(|| PathBuf::from("dub_work").join("_tmp"));
    let input = VideoInput::open(&a.source, &tmp_work).context("abrir vídeo")?;
    let work_dir = a.work_dir.clone().unwrap_or_else(|| PathBuf::from("dub_work").join(sanitize(&input.title)));
    let input = input.with_work_dir(&work_dir)?;
    eprintln!("▶ vídeo: {}", input.title);

    let job = input.prepare().context("extrair áudio")?;

    // 2. Engines ----------------------------------------------------------
    eprintln!("▶ carregando engines…");
    let choice = engines::EngineChoice {
        transcriber_impl: a.transcriber,
        whisper_model: a.whisper_model,
        translator: a.translator,
        translator_model: a.translator_model,
        tts: a.tts,
        tts_impl: a.tts_impl,
        voice: a.voice,
        separate: !a.no_separate,
        diarize: !a.no_diarize,
        mock: a.mock,
    };
    let built = engines::build(&choice, &backend, &work_dir)?;
    for (role, info) in built.engines.describe() {
        let kind = if built.native_roles.contains(&role) { "nativa" } else { "python" };
        eprintln!("  {role:<12} {} [{}] em {} ({kind})", info.name, info.model, info.device);
    }
    let engine_set = built.engines;

    // 3. Pipeline ---------------------------------------------------------
    let cfg = PipelineConfig {
        source_lang: if a.from.is_empty() { None } else { Some(Lang::from(a.from.as_str())) },
        target_lang: Lang::from(a.to.as_str()),
        background_gain: a.background_gain,
        timing: dub_core::timing::TimingConfig { max_speed: a.max_speed, ..Default::default() },
        ..Default::default()
    };

    let stretcher = engines::default_stretcher(a.ffmpeg_stretch)?;
    let pipeline = Pipeline::new(engine_set, cfg).with_stretcher(stretcher);

    let (tx, rx) = crossbeam_channel::unbounded::<Progress>();
    let ui = std::thread::spawn(move || progress_ui(rx));
    let output = pipeline.run(&job, Some(tx)).context("pipeline de dublagem")?;
    let _ = ui.join();

    // 4. Saída ------------------------------------------------------------
    let out_path = a.out.unwrap_or_else(|| input.default_output(&a.to));
    let opts = VideoOptions { keep_original_audio: !a.no_original_track, embed_subtitles: !a.no_subtitles, ..Default::default() };
    let final_path = input.finish(&output, &out_path, &opts).context("remux")?;

    let secs = started.elapsed().as_secs();
    eprintln!("\n✔ pronto em {}m{:02}s", secs / 60, secs % 60);
    eprintln!("  vídeo:    {}", final_path.display());
    eprintln!("  legenda:  {}", output.srt.display());
    eprintln!("  segmentos: {}", output.segments.len());
    let sped: Vec<_> = output.segments.iter().filter(|s| s.speed_factor > 1.25).collect();
    if !sped.is_empty() {
        eprintln!("  ⚠ {} segmentos acelerados acima de 1.25x (tradução longa)", sped.len());
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

fn sanitize(s: &str) -> String {
    let c: String = s.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    c.chars().take(60).collect()
}
