//! Montagem do `EngineSet`: para cada papel tenta a engine **nativa** (Rust/C++)
//! e, se ela não estiver disponível (modelo ausente, lib não encontrada, feature
//! desligada), cai para a engine **Python**. O usuário pode forçar uma delas.

use anyhow::{anyhow, Context, Result};
use dub_core::{Backend, Device, EngineSet, Synthesizer, Transcriber};
use dub_engines_py::{PyDiarizer, PySeparator, PySpec, PySynthesizer, PyTranscriber, PyTranslator, PythonEnv};
use std::path::Path;
use std::sync::Arc;
use tracing::{info, warn};

/// Preferência de implementação por papel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Impl {
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

#[derive(Debug, Clone)]
pub struct EngineChoice {
    pub transcriber_impl: Impl,
    pub whisper_model: String,
    pub translator: String,
    pub translator_model: Option<String>,
    pub tts: String,
    pub tts_impl: Impl,
    pub voice: Option<String>,
    pub separate: bool,
    pub diarize: bool,
    pub mock: bool,
}

pub struct Built {
    pub engines: EngineSet,
    /// Quais papéis ficaram nativos (para o relatório da CLI).
    pub native_roles: Vec<&'static str>,
}

pub fn build(choice: &EngineChoice, backend: &Backend, work_dir: &Path) -> Result<Built> {
    let gpu = backend.preferred;
    let cpu_threads = backend.cpu_threads_for_side_work();
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
            PyTranscriber::spawn(&env, PySpec::new("whisper").model(&choice.whisper_model).device(gpu).threads(backend.cpu_threads))
                .context("engine de transcrição faster-whisper")?,
        ))
    };
    let transcriber: Arc<dyn Transcriber> = match choice.transcriber_impl {
        Impl::Python => py_transcriber(&mut py_env)?,
        Impl::Native => native_transcriber(choice, gpu, backend.cpu_threads)
            .context("transcrição nativa (whisper.cpp)")?
            .ok_or_else(|| anyhow!("transcrição nativa indisponível: compile com a feature 'native'"))
            .inspect(|_| native_roles.push("transcriber"))?,
        Impl::Auto => match native_transcriber(choice, gpu, backend.cpu_threads) {
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

    // --- Tradução (Python por enquanto; CTranslate2 nativo na Parte 2b) ------
    // Fica na CPU de propósito para rodar em paralelo com a síntese na GPU.
    let env = py(&mut py_env)?;
    let mut tr_spec = PySpec::new(&choice.translator).device(Device::Cpu).threads(cpu_threads);
    if let Some(m) = &choice.translator_model {
        tr_spec = tr_spec.model(m);
    }
    let translator = PyTranslator::spawn(&env, tr_spec).with_context(|| format!("engine de tradução '{}'", choice.translator))?;

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
        match PySeparator::spawn(&env, PySpec::new("demucs").device(gpu)) {
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
        match PyDiarizer::spawn(&env, PySpec::new("pyannote").device(gpu)) {
            Ok(d) => Some(Arc::new(d) as Arc<dyn dub_core::Diarizer>),
            Err(e) => {
                warn!("diarização desligada: {e}");
                None
            }
        }
    } else {
        None
    };

    Ok(Built { engines: EngineSet { transcriber, translator: Arc::new(translator), synthesizer, separator, diarizer }, native_roles })
}

fn python_tts(env: &PythonEnv, choice: &EngineChoice, gpu: Device, work_dir: &Path) -> Result<Arc<dyn Synthesizer>> {
    let mut spec = PySpec::new(&choice.tts).device(gpu);
    match (&choice.tts[..], &choice.voice) {
        ("piper", None) => return Err(anyhow!("a engine piper precisa de --voice <nome ou caminho .onnx>")),
        (_, Some(v)) => spec = spec.model(v),
        _ => {}
    }
    Ok(Arc::new(PySynthesizer::spawn(env, spec, &work_dir.join("tts_tmp")).with_context(|| format!("engine de síntese '{}'", choice.tts))?))
}

// --- nativas (compiladas só com a feature `native`) ------------------------------

#[cfg(feature = "native")]
fn native_transcriber(choice: &EngineChoice, gpu: Device, threads: usize) -> Result<Option<Arc<dyn Transcriber>>> {
    let t = dub_native::WhisperTranscriber::load(&choice.whisper_model, gpu, threads)?;
    Ok(Some(Arc::new(t)))
}

#[cfg(not(feature = "native"))]
fn native_transcriber(_: &EngineChoice, _: Device, _: usize) -> Result<Option<Arc<dyn Transcriber>>> {
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
