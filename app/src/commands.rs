//! Comandos chamados pelo frontend (`invoke`). Operações longas rodam em thread
//! própria e reportam por eventos: `job-event`, `models-progress`, `models-done`.

use crate::settings::{self, Settings};
use crate::state::{AppState, RunningJob};
use dub_runtime::{DoctorReport, EngineChoice, JobEvent, JobRequest};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_dialog::DialogExt;
use tauri_plugin_opener::OpenerExt;

type CmdResult<T> = Result<T, String>;

#[tauri::command]
pub fn doctor() -> DoctorReport {
    dub_runtime::doctor()
}

#[tauri::command]
pub fn list_models() -> Vec<dub_runtime::doctor::ModelStatus> {
    dub_runtime::doctor().models
}

#[tauri::command]
pub fn voices_available() -> Vec<String> {
    dub_runtime::doctor().models.into_iter().filter(|m| m.kind == "voices" && m.installed).map(|m| m.name).collect()
}

#[derive(Debug, Clone, Serialize)]
struct ModelsProgress {
    name: String,
    file: String,
    downloaded: u64,
    total: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
struct ModelsDone {
    ok: bool,
    message: String,
}

/// Baixa os modelos pedidos (ou o conjunto padrão se a lista vier vazia).
#[tauri::command]
pub fn download_models(app: AppHandle, state: State<'_, AppState>, names: Vec<String>) -> CmdResult<()> {
    if state.downloading.swap(true, Ordering::SeqCst) {
        return Err("já existe um download em andamento".into());
    }
    let entries: Vec<dub_models::ModelEntry> = if names.is_empty() {
        dub_models::defaults()
    } else {
        names
            .iter()
            .map(|n| dub_models::find(n).ok_or_else(|| format!("'{n}' não está no catálogo")))
            .collect::<Result<Vec<_>, _>>()
            .inspect_err(|_| state.downloading.store(false, Ordering::SeqCst))?
    };

    std::thread::Builder::new()
        .name("dub-models".into())
        .spawn(move || {
            let mut failed = Vec::new();
            for m in &entries {
                let name = m.name.clone();
                let app2 = app.clone();
                let r = dub_models::install(m, move |p| {
                    let _ = app2.emit(
                        "models-progress",
                        ModelsProgress { name: name.clone(), file: p.file, downloaded: p.downloaded, total: p.total },
                    );
                });
                if let Err(e) = r {
                    failed.push(format!("{}: {e}", m.name));
                }
            }
            let st = app.state::<AppState>();
            st.downloading.store(false, Ordering::SeqCst);
            let _ = app.emit(
                "models-done",
                ModelsDone {
                    ok: failed.is_empty(),
                    message: if failed.is_empty() { "modelos instalados".into() } else { failed.join("\n") },
                },
            );
        })
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Parâmetros que a tela de dublagem manda (o resto vem das configurações).
#[derive(Debug, Clone, Deserialize)]
pub struct StartArgs {
    pub source: String,
    pub voice: Option<String>,
    pub whisper_model: Option<String>,
    pub output_dir: Option<PathBuf>,
}

#[tauri::command]
pub fn start_dub(app: AppHandle, state: State<'_, AppState>, args: StartArgs) -> CmdResult<()> {
    {
        let job = state.job.lock().map_err(|_| "estado inconsistente")?;
        if job.is_some() {
            return Err("já existe uma dublagem em andamento".into());
        }
    }
    let s = state.settings.lock().map_err(|_| "estado inconsistente")?.clone();
    let source = args.source.trim().to_string();
    if source.is_empty() {
        return Err("informe um arquivo ou URL".into());
    }

    let is_url = source.starts_with("http://") || source.starts_with("https://");
    let output_dir = args.output_dir.or(s.output_dir.clone()).or_else(|| if is_url { dirs_videos() } else { None });
    let req = JobRequest {
        source,
        output: None,
        output_dir,
        work_dir: Some(work_root(&app).join(dub_runtime::job::sanitize(&chrono_like_stamp()))),
        source_lang: if s.source_lang.is_empty() { None } else { Some(s.source_lang.clone()) },
        target_lang: s.target_lang.clone(),
        force_cpu: s.force_cpu,
        performance: s.performance,
        device_mode: s.device_mode,
        engines: EngineChoice {
            whisper_model: args.whisper_model.unwrap_or(s.whisper_model.clone()),
            voice: Some(args.voice.unwrap_or(s.voice.clone())),
            translator_model: s.translator_model.clone(),
            separate: s.separate,
            separator_model: None,
            voice_male: None,
            voice_female: None,
            diarize: s.diarize,
            ..EngineChoice::default()
        },
        keep_original_audio: s.keep_original_audio,
        embed_subtitles: s.embed_subtitles,
        background_gain: s.background_gain,
        max_speed: s.max_speed,
        ffmpeg_stretch: false,
        content: s.content,
        subtitles: None,
        online_subtitles: s.online_subtitles,
    };

    let cancel = Arc::new(AtomicBool::new(false));
    *state.job.lock().map_err(|_| "estado inconsistente")? = Some(RunningJob { cancel: cancel.clone() });

    let open_when_done = s.open_when_done;
    std::thread::Builder::new()
        .name("dub-job".into())
        .spawn(move || {
            let app_ev = app.clone();
            let result = dub_runtime::run_video_job(&req, cancel, |ev| {
                let _ = app_ev.emit("job-event", &ev);
            });
            match result {
                Ok(r) => {
                    if open_when_done {
                        if let Some(dir) = r.video.parent() {
                            let _ = app.opener().reveal_item_in_dir(dir.join(r.video.file_name().unwrap_or_default()));
                        }
                    }
                }
                Err(e) => {
                    let _ = app.emit("job-event", JobEvent::Error { message: format!("{e:#}") });
                }
            }
            if let Ok(mut j) = app.state::<AppState>().job.lock() {
                *j = None;
            }
        })
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn cancel_dub(state: State<'_, AppState>) -> CmdResult<()> {
    let job = state.job.lock().map_err(|_| "estado inconsistente")?;
    match job.as_ref() {
        Some(j) => {
            j.cancel.store(true, Ordering::Relaxed);
            Ok(())
        }
        None => Err("nenhuma dublagem em andamento".into()),
    }
}

#[tauri::command]
pub fn job_running(state: State<'_, AppState>) -> bool {
    state.job.lock().map(|j| j.is_some()).unwrap_or(false)
}

#[tauri::command]
pub fn get_settings(state: State<'_, AppState>) -> CmdResult<Settings> {
    state.settings.lock().map(|s| s.clone()).map_err(|_| "estado inconsistente".into())
}

#[tauri::command]
pub fn save_settings(app: AppHandle, state: State<'_, AppState>, settings: Settings) -> CmdResult<()> {
    settings::save(&app, &settings)?;
    *state.settings.lock().map_err(|_| "estado inconsistente")? = settings;
    Ok(())
}

#[tauri::command]
pub fn pick_video(app: AppHandle) -> Option<PathBuf> {
    app.dialog()
        .file()
        .add_filter("Vídeos", &["mp4", "mkv", "mov", "webm", "avi", "m4v"])
        .blocking_pick_file()
        .and_then(|f| f.into_path().ok())
}

#[tauri::command]
pub fn pick_folder(app: AppHandle) -> Option<PathBuf> {
    app.dialog().file().blocking_pick_folder().and_then(|f| f.into_path().ok())
}

#[tauri::command]
pub fn open_path(app: AppHandle, path: String) -> CmdResult<()> {
    app.opener().open_path(path, None::<&str>).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn reveal_path(app: AppHandle, path: String) -> CmdResult<()> {
    app.opener().reveal_item_in_dir(path).map_err(|e| e.to_string())
}

// --- helpers --------------------------------------------------------------------

/// Pasta de trabalho do app (arquivos intermediários), fora da pasta de saída.
fn work_root(app: &AppHandle) -> PathBuf {
    app.path().app_cache_dir().unwrap_or_else(|_| std::env::temp_dir().join("dub")).join("work")
}

fn dirs_videos() -> Option<PathBuf> {
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))?;
    let v = PathBuf::from(&home).join("Videos");
    Some(if v.is_dir() { v } else { PathBuf::from(home) })
}

/// Carimbo simples para nomear a pasta de trabalho sem puxar o crate chrono.
fn chrono_like_stamp() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("job_{secs}")
}
