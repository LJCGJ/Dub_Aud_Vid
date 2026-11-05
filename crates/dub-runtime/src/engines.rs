//! Montagem do `EngineSet`: para cada papel tenta a engine **nativa** (Rust/C++)
//! e, se ela não estiver disponível (modelo ausente, lib não encontrada, feature
//! desligada), cai para a engine **Python**. O usuário pode forçar uma delas.

use anyhow::{anyhow, Context, Result};
use dub_core::{Backend, Device, EngineSet, Role, Synthesizer, Transcriber, Translator};
use dub_engines_py::{PyDiarizer, PySeparator, PySpec, PySynthesizer, PyTranscriber, PyTranslator, PythonEnv};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use tracing::{info, warn};

/// Preferência de implementação por papel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Impl {
    #[default]
    /// Nativa se possível, senão Python.
    Auto,
    Native,
    Python,
}

impl std::str::FromStr for Impl {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s {
            "auto" => Ok(Impl::Auto),
            "native" | "nativa" => Ok(Impl::Native),
            "python" | "py" => Ok(Impl::Python),
            _ => Err(format!("valor inválido '{s}' (auto | native | python)")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EngineChoice {
    pub transcriber_impl: Impl,
    pub whisper_model: String,
    pub translator_impl: Impl,
    pub translator_model: Option<String>,
    pub tts: String,
    pub tts_impl: Impl,
    pub voice: Option<String>,
    pub separate: bool,
    pub diarize: bool,
    pub mock: bool,
}

impl Default for EngineChoice {
    fn default() -> Self {
        Self {
            transcriber_impl: Impl::Auto,
            whisper_model: "medium".into(),
            translator_impl: Impl::Auto,
            translator_model: None,
            tts: "piper".into(),
            tts_impl: Impl::Auto,
            voice: Some("pt_BR-faber-medium".into()),
            separate: false,
            diarize: false,
            mock: false,
        }
    }
}

pub struct Built {
    pub engines: EngineSet,
    /// Quais papéis ficaram nativos (para o relatório da CLI).
    pub native_roles: Vec<&'static str>,
}

pub fn build(choice: &EngineChoice, backend: &Backend, work_dir: &Path) -> Result<Built> {
    let gpu = backend.device_for(Role::Synthesize);
    let transcribe_dev = backend.device_for(Role::Transcribe);
    let separate_dev = backend.device_for(Role::Separate);
    let cpu_threads = backend.cpu_threads_for_side_work();
    let work_threads = backend.work_threads();
    let mut native_roles = Vec::new();

    // Python é opcional: só é resolvido se alguma engine precisar dele.
    let mut py_env: Option<PythonEnv> = None;
    let py = |py_env: &mut Option<PythonEnv>| -> Result<PythonEnv> {
        if py_env.is_none() {
            *py_env = Some(PythonEnv::discover(None).context("ambiente Python das engines (py/.venv)")?);
        }
        Ok(py_env.clone().unwrap())
    };

    if choice.mock {
        let env = py(&mut py_env)?;
        let m = || PySpec::new("mock");
        return Ok(Built {
            engines: EngineSet {
                transcriber: Arc::new(PyTranscriber::spawn(&env, m())?),
                translator: Arc::new(PyTranslator::spawn(&env, m())?),
                synthesizer: Arc::new(PySynthesizer::spawn(&env, m(), &work_dir.join("tts_tmp"))?),
                separator: if choice.separate { Some(Arc::new(PySeparator::spawn(&env, m())?)) } else { None },
                diarizer: if choice.diarize { Some(Arc::new(PyDiarizer::spawn(&env, m())?)) } else { None },
            },
            native_roles,
        });
    }

    // --- Transcrição -------------------------------------------------------
    let py_transcriber = |py_env: &mut Option<PythonEnv>| -> Result<Arc<dyn Transcriber>> {
        let env = py(py_env)?;
        Ok(Arc::new(
            PyTranscriber::spawn(
                &env,
                PySpec::new("whisper").model(&choice.whisper_model).device(transcribe_dev).threads(backend.cpu_threads),
            )
            .context("engine de transcrição faster-whisper")?,
        ))
    };
    let transcriber: Arc<dyn Transcriber> = match choice.transcriber_impl {
        Impl::Python => py_transcriber(&mut py_env)?,
        Impl::Native => native_transcriber(choice, backend, work_threads)
            .context("transcrição nativa (whisper.cpp)")?
            .ok_or_else(|| anyhow!("transcrição nativa indisponível: compile com a feature 'native'"))
            .inspect(|_| native_roles.push("transcriber"))?,
        Impl::Auto => match native_transcriber(choice, backend, work_threads) {
            Ok(Some(t)) => {
                native_roles.push("transcriber");
                t
            }
            Ok(None) => py_transcriber(&mut py_env)?,
            Err(e) => {
                info!("whisper.cpp indisponível ({e}); usando faster-whisper em Python");
                py_transcriber(&mut py_env)?
            }
        },
    };

    // --- Tradução ----------------------------------------------------------
    // Fica na CPU de propósito para rodar em paralelo com a síntese na GPU.
    let py_translator = |py_env: &mut Option<PythonEnv>| -> Result<Arc<dyn Translator>> {
        let env = py(py_env)?;
        let engine = match choice.translator_model.as_deref() {
            Some("nllb") => "nllb",
            Some("marian") | None => "marian",
            Some(other) if other.contains('/') => "marian", // id do Hugging Face → Marian com modelo custom
            Some(_) => "marian",
        };
        let mut spec = PySpec::new(engine).device(Device::Cpu).threads(cpu_threads);
        if let Some(m) = choice.translator_model.as_deref().filter(|m| m.contains('/')) {
            spec = spec.model(m);
        }
        Ok(Arc::new(PyTranslator::spawn(&env, spec).with_context(|| format!("engine de tradução '{engine}'"))?))
    };
    let translator: Arc<dyn Translator> = match choice.translator_impl {
        Impl::Python => py_translator(&mut py_env)?,
        Impl::Native => native_translator(choice, cpu_threads)
            .context("tradução nativa (CTranslate2)")?
            .ok_or_else(|| anyhow!("tradução nativa indisponível: compile com a feature 'native'"))
            .inspect(|_| native_roles.push("translator"))?,
        Impl::Auto => match native_translator(choice, cpu_threads) {
            Ok(Some(t)) => {
                native_roles.push("translator");
                t
            }
            Ok(None) => py_translator(&mut py_env)?,
            Err(e) => {
                info!("CTranslate2 indisponível ({e}); usando transformers em Python");
                py_translator(&mut py_env)?
            }
        },
    };

    // --- Síntese -----------------------------------------------------------
    let synthesizer: Arc<dyn Synthesizer> = if choice.tts == "piper" {
        match choice.tts_impl {
            Impl::Python => python_tts(&py(&mut py_env)?, choice, gpu, work_dir)?,
            Impl::Native => native_piper(choice, gpu, cpu_threads)
                .context("síntese nativa (Piper/ONNX Runtime)")?
                .ok_or_else(|| anyhow!("síntese nativa indisponível: compile com a feature 'native'"))
                .inspect(|_| native_roles.push("synthesizer"))?,
            Impl::Auto => match native_piper(choice, gpu, cpu_threads) {
                Ok(Some(s)) => {
                    native_roles.push("synthesizer");
                    s
                }
                Ok(None) => python_tts(&py(&mut py_env)?, choice, gpu, work_dir)?,
                Err(e) => {
                    info!("Piper nativo indisponível ({e}); usando Piper em Python");
                    python_tts(&py(&mut py_env)?, choice, gpu, work_dir)?
                }
            },
        }
    } else {
        python_tts(&py(&mut py_env)?, choice, gpu, work_dir)?
    };

    // --- Opcionais: falha neles não derruba o trabalho, só avisa -----------
    let separator = if choice.separate {
        match py(&mut py_env)
            .and_then(|env| PySeparator::spawn(&env, PySpec::new("demucs").device(separate_dev)).map_err(Into::into))
        {
            Ok(s) => Some(Arc::new(s) as Arc<dyn dub_core::Separator>),
            Err(e) => {
                warn!("separação desligada: {e}");
                None
            }
        }
    } else {
        None
    };
    let diarizer = if choice.diarize {
        match py(&mut py_env)
            .and_then(|env| PyDiarizer::spawn(&env, PySpec::new("pyannote").device(separate_dev)).map_err(Into::into))
        {
            Ok(d) => Some(Arc::new(d) as Arc<dyn dub_core::Diarizer>),
            Err(e) => {
                warn!("diarização desligada: {e}");
                None
            }
        }
    } else {
        None
    };

    Ok(Built { engines: EngineSet { transcriber, translator, synthesizer, separator, diarizer }, native_roles })
}

fn python_tts(env: &PythonEnv, choice: &EngineChoice, gpu: Device, work_dir: &Path) -> Result<Arc<dyn Synthesizer>> {
    let mut spec = PySpec::new(&choice.tts).device(gpu);
    match (&choice.tts[..], &choice.voice) {
        ("piper", None) => return Err(anyhow!("a engine piper precisa de --voice <nome ou caminho .onnx>")),
        (_, Some(v)) => spec = spec.model(v),
        _ => {}
    }
    Ok(Arc::new(
        PySynthesizer::spawn(env, spec, &work_dir.join("tts_tmp"))
            .with_context(|| format!("engine de síntese '{}'", choice.tts))?,
    ))
}

// --- nativas (compiladas só com a feature `native`) ------------------------------

#[cfg(feature = "native")]
fn native_transcriber(choice: &EngineChoice, backend: &Backend, threads: usize) -> Result<Option<Arc<dyn Transcriber>>> {
    let device = backend.device_for(Role::Transcribe);
    let (model, threads) = fit_whisper_to_ram(&choice.whisper_model, device, backend, threads);
    let t = dub_native::WhisperTranscriber::load_with_backend(&model, device, threads, Some(backend))?;
    Ok(Some(Arc::new(t)))
}

/// Mantém a máquina confortável: se o modelo pedido não cabe na RAM livre com
/// folga, usa o maior modelo menor que já esteja instalado e avisa; se nenhum
/// couber, segue com o pedido mas avisa que pode ficar lento (swap).
#[cfg(feature = "native")]
fn fit_whisper_to_ram(requested: &str, device: Device, backend: &Backend, threads: usize) -> (String, usize) {
    use dub_native::WhisperTranscriber as W;
    let size_of = |m: &str| W::model_size(m).unwrap_or(0);
    let Some(headroom) = backend.ram_headroom_mb(size_of(requested)) else { return (requested.to_string(), threads) };
    if headroom >= 0 {
        return (requested.to_string(), threads);
    }
    // Em GPU dedicada o modelo mora na VRAM; a RAM só é problema em CPU/iGPU.
    if device.is_gpu() && !backend.gpu_integrated {
        return (requested.to_string(), threads);
    }
    const LADDER: [&str; 6] = ["large-v3-turbo", "medium", "small", "base", "tiny", "tiny"];
    let start = LADDER.iter().position(|m| requested.starts_with(m)).unwrap_or(0);
    for m in &LADDER[start + 1..] {
        if W::model_available(m) && backend.ram_headroom_mb(size_of(m)).unwrap_or(0) >= 0 {
            warn!(
                requested,
                using = m,
                ram_free_mb = backend.ram_free_mb.unwrap_or(0),
                "pouca RAM livre para o modelo pedido; usando um menor para a máquina não travar (feche programas ou passe --whisper-model para forçar)"
            );
            return (m.to_string(), threads);
        }
    }
    warn!(
        requested,
        ram_free_mb = backend.ram_free_mb.unwrap_or(0),
        faltam_mb = -headroom,
        "pouca RAM livre para este modelo e nenhum menor instalado: pode ficar lento (swap). Feche programas ou rode `dub models download small`"
    );
    // Menos threads = menos memória de trabalho e menos pressão.
    (requested.to_string(), (threads / 2).max(1))
}

#[cfg(not(feature = "native"))]
fn native_transcriber(_: &EngineChoice, _: &Backend, _: usize) -> Result<Option<Arc<dyn Transcriber>>> {
    Ok(None)
}

pub const DEFAULT_CT2_MODEL: &str = "opus-mt-tc-big-en-pt-ct2-int8";

#[cfg(feature = "native")]
fn native_translator(choice: &EngineChoice, threads: usize) -> Result<Option<Arc<dyn Translator>>> {
    let model = match choice.translator_model.as_deref() {
        // Nomes das engines Python não são modelos CT2: deixa o fallback decidir.
        None | Some("marian") => DEFAULT_CT2_MODEL,
        Some("nllb") => return Err(anyhow!("nllb só existe na engine Python")),
        Some(m) => m,
    };
    let t = dub_native::Ct2Translator::load(model, Device::Cpu, threads)?;
    Ok(Some(Arc::new(t)))
}

#[cfg(not(feature = "native"))]
fn native_translator(_: &EngineChoice, _: usize) -> Result<Option<Arc<dyn Translator>>> {
    Ok(None)
}

#[cfg(feature = "native")]
fn native_piper(choice: &EngineChoice, gpu: Device, threads: usize) -> Result<Option<Arc<dyn Synthesizer>>> {
    let voice = choice.voice.as_deref().ok_or_else(|| anyhow!("a engine piper precisa de --voice <nome ou caminho .onnx>"))?;
    let s = dub_native::PiperSynthesizer::load(voice, gpu, threads)?;
    Ok(Some(Arc::new(s)))
}

#[cfg(not(feature = "native"))]
fn native_piper(_: &EngineChoice, _: Device, _: usize) -> Result<Option<Arc<dyn Synthesizer>>> {
    Ok(None)
}

/// Stretcher padrão: WSOLA em C++ quando a feature `native` está ligada, senão ffmpeg.
pub fn default_stretcher(prefer_ffmpeg: bool) -> Result<Arc<dyn dub_core::pipeline::Stretcher>> {
    #[cfg(feature = "native")]
    {
        if !prefer_ffmpeg {
            return Ok(Arc::new(dub_native::NativeStretcher));
        }
    }
    let ff = dub_media::Ffmpeg::locate()?;
    Ok(Arc::new(dub_media::FfmpegStretcher::new(ff)?))
}
