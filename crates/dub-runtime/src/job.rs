//! Execução de uma dublagem de vídeo ponta a ponta, com eventos para a UI.

use crate::engines::{self, EngineChoice};
use anyhow::{Context, Result};
use dub_core::engines::EngineInfo;
use dub_core::{Backend, Lang, Pipeline, PipelineConfig, Progress, Segment};
use dub_input_video::{VideoInput, VideoOptions};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Tudo que um job precisa. Serializável para a UI mandar como JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRequest {
    /// Caminho local ou URL.
    pub source: String,
    /// Arquivo de saída completo. Tem prioridade sobre `output_dir`.
    pub output: Option<PathBuf>,
    /// Pasta de saída; o nome vira `<título>.dub.<idioma>.mp4`. `None` = ao lado
    /// do original (ou pasta atual, para URL).
    pub output_dir: Option<PathBuf>,
    /// Pasta de trabalho; `None` = `dub_work/<título>`.
    pub work_dir: Option<PathBuf>,
    pub source_lang: Option<String>,
    pub target_lang: String,
    pub force_cpu: bool,
    pub engines: EngineChoice,
    pub keep_original_audio: bool,
    pub embed_subtitles: bool,
    pub background_gain: f32,
    pub max_speed: f32,
    pub ffmpeg_stretch: bool,
}

impl Default for JobRequest {
    fn default() -> Self {
        Self {
            source: String::new(),
            output: None,
            output_dir: None,
            work_dir: None,
            source_lang: Some("en".into()),
            target_lang: "pt-BR".into(),
            force_cpu: false,
            engines: EngineChoice::default(),
            keep_original_audio: true,
            embed_subtitles: true,
            background_gain: 0.9,
            max_speed: 1.35,
            ffmpeg_stretch: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineLine {
    pub role: String,
    pub name: String,
    pub model: String,
    pub device: String,
    pub native: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobResult {
    pub video: PathBuf,
    pub srt: PathBuf,
    pub work_dir: PathBuf,
    pub title: String,
    pub segments: Vec<Segment>,
    pub elapsed_secs: u64,
    /// Segmentos acelerados acima de 1.25x (tradução longa demais para a janela).
    pub rushed_segments: usize,
}

/// Eventos emitidos durante o job, na ordem em que acontecem.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum JobEvent {
    /// Mensagem de status livre ("baixando vídeo", "carregando engines"...).
    Status {
        message: String,
    },
    /// Dispositivo escolhido.
    Device {
        device: String,
        gpu_name: Option<String>,
    },
    /// Vídeo aberto.
    Opened {
        title: String,
        work_dir: PathBuf,
    },
    /// Engines carregadas.
    Engines {
        engines: Vec<EngineLine>,
    },
    /// Progresso de uma etapa do pipeline.
    Progress(Progress),
    Done(JobResult),
    Error {
        message: String,
    },
}

/// Executa o job inteiro na thread atual. `on_event` recebe cada evento; `cancel`
/// pode ser setado de outra thread para interromper no próximo ponto seguro.
pub fn run_video_job(req: &JobRequest, cancel: Arc<AtomicBool>, mut on_event: impl FnMut(JobEvent)) -> Result<JobResult> {
    let started = Instant::now();
    let backend = if req.force_cpu { Backend::cpu_only() } else { Backend::detect() };
    on_event(JobEvent::Device { device: backend.preferred.to_string(), gpu_name: backend.gpu_name.clone() });

    // 1. Entrada ----------------------------------------------------------
    on_event(JobEvent::Status {
        message: if dub_media::YtDlp::is_url(&req.source) { "baixando vídeo".into() } else { "abrindo vídeo".into() },
    });
    let tmp_work = req.work_dir.clone().unwrap_or_else(|| PathBuf::from("dub_work").join("_tmp"));
    let input = VideoInput::open(&req.source, &tmp_work).context("abrir vídeo")?;
    let work_dir = req.work_dir.clone().unwrap_or_else(|| PathBuf::from("dub_work").join(sanitize(&input.title)));
    let input = input.with_work_dir(&work_dir)?;
    on_event(JobEvent::Opened { title: input.title.clone(), work_dir: work_dir.clone() });
    check_cancel(&cancel)?;

    on_event(JobEvent::Status { message: "extraindo áudio".into() });
    let job = input.prepare().context("extrair áudio")?;
    check_cancel(&cancel)?;

    // 2. Engines ----------------------------------------------------------
    on_event(JobEvent::Status { message: "carregando engines".into() });
    let built = engines::build(&req.engines, &backend, &work_dir)?;
    let lines = built
        .engines
        .describe()
        .into_iter()
        .map(|(role, info): (&str, EngineInfo)| EngineLine {
            role: role.to_string(),
            name: info.name,
            model: info.model,
            device: info.device,
            native: built.native_roles.contains(&role),
        })
        .collect();
    on_event(JobEvent::Engines { engines: lines });
    check_cancel(&cancel)?;

    // 3. Pipeline ---------------------------------------------------------
    let cfg = PipelineConfig {
        source_lang: req.source_lang.as_deref().filter(|s| !s.is_empty()).map(Lang::from),
        target_lang: Lang::from(req.target_lang.as_str()),
        background_gain: req.background_gain,
        timing: dub_core::timing::TimingConfig { max_speed: req.max_speed, ..Default::default() },
        ..Default::default()
    };
    let stretcher = engines::default_stretcher(req.ffmpeg_stretch)?;
    let pipeline = Pipeline::new(built.engines, cfg).with_stretcher(stretcher);

    // Propaga o cancelamento externo para o pipeline.
    let pipe_cancel = pipeline.cancel_handle();
    let ext_cancel = cancel.clone();
    let watcher_stop = Arc::new(AtomicBool::new(false));
    let watcher_stop2 = watcher_stop.clone();
    let watcher = std::thread::spawn(move || {
        while !watcher_stop2.load(Ordering::Relaxed) {
            if ext_cancel.load(Ordering::Relaxed) {
                pipe_cancel.store(true, Ordering::Relaxed);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    });

    let (tx, rx) = crossbeam_channel::unbounded::<Progress>();
    let run = std::thread::scope(|s| {
        let h = s.spawn(|| pipeline.run(&job, Some(tx)));
        for p in rx.iter() {
            on_event(JobEvent::Progress(p));
        }
        h.join().expect("pipeline thread")
    });
    watcher_stop.store(true, Ordering::Relaxed);
    let _ = watcher.join();
    let output = run.context("pipeline de dublagem")?;

    // 4. Saída ------------------------------------------------------------
    on_event(JobEvent::Status { message: "gerando vídeo final".into() });
    let default_out = input.default_output(&req.target_lang);
    let out_path = match (&req.output, &req.output_dir) {
        (Some(o), _) => o.clone(),
        (None, Some(dir)) => dir.join(default_out.file_name().unwrap_or_default()),
        (None, None) => default_out,
    };
    let opts =
        VideoOptions { keep_original_audio: req.keep_original_audio, embed_subtitles: req.embed_subtitles, ..Default::default() };
    let final_path = input.finish(&output, &out_path, &opts).context("remux")?;

    let result = JobResult {
        video: final_path,
        srt: output.srt.clone(),
        work_dir,
        title: input.title.clone(),
        rushed_segments: output.segments.iter().filter(|s| s.speed_factor > 1.25).count(),
        segments: output.segments,
        elapsed_secs: started.elapsed().as_secs(),
    };
    on_event(JobEvent::Done(result.clone()));
    Ok(result)
}

fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        anyhow::bail!("cancelado");
    }
    Ok(())
}

pub fn sanitize(s: &str) -> String {
    let c: String = s.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    c.chars().take(60).collect()
}
