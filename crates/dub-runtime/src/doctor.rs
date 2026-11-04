//! Diagnóstico: o que a máquina tem e o que falta. Usado por `dub doctor` e pela
//! tela de configurações do app.

use dub_core::paths::ModelKind;
use dub_core::Backend;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolStatus {
    pub name: String,
    pub ok: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatus {
    pub kind: String,
    pub name: String,
    pub installed: bool,
    pub path: Option<String>,
    pub default: bool,
    pub size_mb: u32,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoctorReport {
    pub device: String,
    pub devices: Vec<String>,
    pub cpu_threads: usize,
    pub gpu_name: Option<String>,
    pub tools: Vec<ToolStatus>,
    pub native: Vec<ToolStatus>,
    pub native_enabled: bool,
    pub models: Vec<ModelStatus>,
    /// Pronto para dublar sem Python: whisper + tradutor + voz instalados e engines nativas ok.
    pub ready_native: bool,
}

pub fn doctor() -> DoctorReport {
    let b = Backend::detect();
    let mut tools = vec![
        tool("ffmpeg", dub_media::Ffmpeg::locate().map(|_| "encontrado".into()).map_err(|e| e.to_string())),
        tool("yt-dlp", dub_media::YtDlp::locate().map(|_| "encontrado".into()).map_err(|e| e.to_string())),
        tool(
            "python",
            dub_engines_py::PythonEnv::discover(None).map(|p| p.interpreter.display().to_string()).map_err(|e| e.to_string()),
        ),
    ];
    tools.iter_mut().for_each(|t| {
        if !t.ok {
            t.detail = t.detail.replace("ferramenta externa não encontrada: ", "não encontrado: ");
        }
    });

    let native = native_status();
    let native_enabled = cfg!(feature = "native");

    let mut models = Vec::new();
    let mut have = std::collections::HashMap::new();
    for (kind, entries) in dub_models::download::summary() {
        for (m, st) in entries {
            let (installed, path) = match st {
                dub_models::InstallStatus::Installed(p) => (true, Some(p.display().to_string())),
                dub_models::InstallStatus::Missing => (false, None),
            };
            if installed {
                *have.entry(kind).or_insert(0usize) += 1;
            }
            let size_mb = m.total_mb();
            models.push(ModelStatus {
                kind: kind.to_string(),
                name: m.name,
                installed,
                path,
                default: m.default,
                size_mb,
                description: m.description,
            });
        }
    }
    let all_models = ModelKind::all().iter().all(|k| have.get(k).copied().unwrap_or(0) > 0);
    let ready_native = native_enabled && native.iter().all(|t| t.ok) && all_models && tools[0].ok;

    DoctorReport {
        device: b.preferred.to_string(),
        devices: b.available.iter().map(|d| d.to_string()).collect(),
        cpu_threads: b.cpu_threads,
        gpu_name: b.gpu_name,
        tools,
        native,
        native_enabled,
        models,
        ready_native,
    }
}

fn tool(name: &str, r: Result<String, String>) -> ToolStatus {
    match r {
        Ok(d) => ToolStatus { name: name.into(), ok: true, detail: d },
        Err(e) => ToolStatus { name: name.into(), ok: false, detail: e },
    }
}

#[cfg(feature = "native")]
fn native_status() -> Vec<ToolStatus> {
    let ort = dub_native::piper::find_ort_lib();
    let esp = dub_native::phonemize::find_espeak_data();
    vec![
        ToolStatus { name: "time-stretch (WSOLA C++)".into(), ok: true, detail: "embutido".into() },
        ToolStatus { name: "whisper.cpp".into(), ok: true, detail: "embutido".into() },
        ToolStatus { name: "CTranslate2".into(), ok: true, detail: "embutido".into() },
        ToolStatus {
            name: "onnxruntime (Piper)".into(),
            ok: ort.is_some(),
            detail: ort
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "não encontrada: coloque onnxruntime ao lado do executável ou defina DUB_ORT_LIB".into()),
        },
        ToolStatus {
            name: "espeak-ng-data (Piper)".into(),
            ok: esp.is_some(),
            detail: esp
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "não encontrada: pasta espeak-ng-data ao lado do executável ou DUB_ESPEAK_DATA".into()),
        },
    ]
}

#[cfg(not(feature = "native"))]
fn native_status() -> Vec<ToolStatus> {
    vec![ToolStatus { name: "engines nativas".into(), ok: false, detail: "desligadas neste build (feature 'native')".into() }]
}
