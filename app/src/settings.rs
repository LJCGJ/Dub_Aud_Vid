//! Preferências persistidas em JSON na pasta de configuração do usuário.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Manager};
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Pasta onde salvar os vídeos dublados (vazio = ao lado do original / Vídeos).
    pub output_dir: Option<PathBuf>,
    pub whisper_model: String,
    pub voice: String,
    pub translator_model: Option<String>,
    pub source_lang: String,
    pub target_lang: String,
    pub keep_original_audio: bool,
    pub embed_subtitles: bool,
    pub background_gain: f32,
    pub max_speed: f32,
    pub force_cpu: bool,
    /// eco | balanced | max
    pub performance: dub_core::Performance,
    /// auto | cpu | gpu | split
    #[serde(default)]
    pub device_mode: dub_core::DeviceMode,
    /// Recursos avançados que exigem as engines Python.
    pub separate: bool,
    pub diarize: bool,
    /// Abrir a pasta ao terminar.
    pub open_when_done: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            output_dir: None,
            whisper_model: "medium".into(),
            voice: "pt_BR-faber-medium".into(),
            translator_model: None,
            source_lang: "en".into(),
            target_lang: "pt-BR".into(),
            keep_original_audio: true,
            embed_subtitles: true,
            background_gain: 0.9,
            max_speed: 1.35,
            force_cpu: false,
            performance: dub_core::Performance::Balanced,
            device_mode: dub_core::DeviceMode::Auto,
            separate: false,
            diarize: false,
            open_when_done: true,
        }
    }
}

fn file(app: &AppHandle) -> Option<PathBuf> {
    app.path().app_config_dir().ok().map(|d| d.join("settings.json"))
}

pub fn load(app: &AppHandle) -> Settings {
    let Some(p) = file(app) else { return Settings::default() };
    match std::fs::read(&p) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|e| {
            warn!("settings.json inválido ({e}); usando padrão");
            Settings::default()
        }),
        Err(_) => Settings::default(),
    }
}

pub fn save(app: &AppHandle, s: &Settings) -> Result<(), String> {
    let p = file(app).ok_or("sem pasta de configuração")?;
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }
    std::fs::write(&p, serde_json::to_vec_pretty(s).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    info!(path = %p.display(), "configurações salvas");
    Ok(())
}

/// Aponta o `dub-native` para os recursos empacotados com o app, se existirem e
/// se o usuário não tiver definido as variáveis manualmente.
pub fn export_resource_env(res: &Path) {
    let set_if_absent = |key: &str, path: PathBuf| {
        if std::env::var_os(key).is_none() && path.exists() {
            std::env::set_var(key, &path);
            info!(key, path = %path.display(), "recurso do bundle");
        }
    };
    set_if_absent("DUB_ESPEAK_DATA", res.join("espeak-ng-data"));
    // ffmpeg/ffprobe/yt-dlp empacotados (Windows). No Linux/macOS vêm do sistema.
    if res.join(format!("ffmpeg{}", std::env::consts::EXE_SUFFIX)).exists() {
        set_if_absent("DUB_TOOLS_DIR", res.to_path_buf());
    }
    let ort = if cfg!(windows) {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    };
    set_if_absent("DUB_ORT_LIB", res.join(ort));
    // Modelos: a pasta de recursos é só leitura em instalações no Program Files;
    // o dub_core::paths cai para %LOCALAPPDATA%\dub quando não consegue gravar.
    if std::env::var_os("DUB_MODELS_DIR").is_none() && res.join("models").is_dir() {
        // Só usa como pasta de modelos se já houver algo nela (bundle com modelos).
        let has_any = ["whisper", "voices", "translate"]
            .iter()
            .any(|k| std::fs::read_dir(res.join("models").join(k)).map(|mut d| d.next().is_some()).unwrap_or(false));
        if has_any {
            std::env::set_var("DUB_MODELS_DIR", res.join("models"));
        }
    }
}
