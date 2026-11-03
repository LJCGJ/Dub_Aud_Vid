//! Montagem do `EngineSet` a partir das opções da CLI.
//!
//! Hoje todas as engines vêm do `dub-engines-py`. Na Parte 2 entram as nativas
//! (`dub-engines`) e esta função passa a tentar nativa → Python → erro.

use anyhow::{anyhow, Context, Result};
use dub_core::{Backend, Device, EngineSet};
use dub_engines_py::{PyDiarizer, PySeparator, PySpec, PySynthesizer, PyTranscriber, PyTranslator, PythonEnv};
use std::path::Path;
use std::sync::Arc;
use tracing::warn;

#[derive(Debug, Clone)]
pub struct EngineChoice {
    pub transcriber: String,
    pub whisper_model: String,
    pub translator: String,
    pub translator_model: Option<String>,
    pub tts: String,
    pub voice: Option<String>,
    pub separate: bool,
    pub diarize: bool,
    pub mock: bool,
}

pub fn build(choice: &EngineChoice, backend: &Backend, work_dir: &Path) -> Result<EngineSet> {
    let env = PythonEnv::discover(None).context("ambiente Python das engines")?;
    let gpu = backend.preferred;
    let cpu_threads = backend.cpu_threads_for_side_work();

    if choice.mock {
        let m = |_: &str| PySpec::new("mock");
        return Ok(EngineSet {
            transcriber: Arc::new(PyTranscriber::spawn(&env, m("t"))?),
            translator: Arc::new(PyTranslator::spawn(&env, m("tr"))?),
            synthesizer: Arc::new(PySynthesizer::spawn(&env, m("s"), &work_dir.join("tts_tmp"))?),
            separator: if choice.separate { Some(Arc::new(PySeparator::spawn(&env, m("sep"))?)) } else { None },
            diarizer: if choice.diarize { Some(Arc::new(PyDiarizer::spawn(&env, m("d"))?)) } else { None },
        });
    }

    // Transcrição: GPU se houver (whisper é a 2ª etapa mais pesada).
    let transcriber = PyTranscriber::spawn(
        &env,
        PySpec::new(&choice.transcriber).model(&choice.whisper_model).device(gpu).threads(backend.cpu_threads),
    )
    .with_context(|| format!("engine de transcrição '{}'", choice.transcriber))?;

    // Tradução: fica na CPU de propósito para rodar em paralelo com a síntese na GPU
    // (modelos Marian/NLLB são leves o bastante para isso).
    let mut tr_spec = PySpec::new(&choice.translator).device(Device::Cpu).threads(cpu_threads);
    if let Some(m) = &choice.translator_model {
        tr_spec = tr_spec.model(m);
    }
    let translator = PyTranslator::spawn(&env, tr_spec).with_context(|| format!("engine de tradução '{}'", choice.translator))?;

    // Síntese: GPU.
    let mut tts_spec = PySpec::new(&choice.tts).device(gpu);
    match (&choice.tts[..], &choice.voice) {
        ("piper", None) => return Err(anyhow!("a engine piper precisa de --voice <caminho/para/voz.onnx>")),
        (_, Some(v)) => tts_spec = tts_spec.model(v),
        _ => {}
    }
    let synthesizer = PySynthesizer::spawn(&env, tts_spec, &work_dir.join("tts_tmp"))
        .with_context(|| format!("engine de síntese '{}'", choice.tts))?;

    // Opcionais: falha neles não derruba o trabalho, só avisa.
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

    Ok(EngineSet {
        transcriber: Arc::new(transcriber),
        translator: Arc::new(translator),
        synthesizer: Arc::new(synthesizer),
        separator,
        diarizer,
    })
}
