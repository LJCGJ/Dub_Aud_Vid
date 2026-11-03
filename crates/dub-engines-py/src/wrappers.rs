//! Adaptadores: `PyProcess` → traits de `dub-core`.

use crate::process::{PyProcess, PySpec};
use crate::python::PythonEnv;
use dub_core::engines::{Diarizer, Engine, EngineInfo, Separator, Synthesizer, Transcriber, Translator};
use dub_core::{AudioBuffer, Lang, Result, Segment, Speaker};
use serde::Deserialize;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, RwLock};

fn info_of(p: &PyProcess) -> EngineInfo {
    EngineInfo { name: p.info.name.clone(), model: p.info.model.clone(), device: p.info.device.clone() }
}

// --- Transcriber ---------------------------------------------------------------

pub struct PyTranscriber(PyProcess);

#[derive(Deserialize)]
struct SegOut {
    start: f64,
    end: f64,
    text: String,
}
#[derive(Deserialize)]
struct TranscribeOut {
    segments: Vec<SegOut>,
}

impl PyTranscriber {
    pub fn spawn(env: &PythonEnv, spec: PySpec) -> Result<Self> {
        Ok(Self(PyProcess::spawn(env, spec)?))
    }
}
impl Engine for PyTranscriber {
    fn info(&self) -> EngineInfo {
        info_of(&self.0)
    }
}
impl Transcriber for PyTranscriber {
    fn transcribe(&self, audio: &Path, lang_hint: Option<&Lang>) -> Result<Vec<Segment>> {
        let out: TranscribeOut = self.0.call("transcribe", json!({ "audio": abs(audio), "lang": lang_hint.map(|l| l.0.clone()) }))?;
        Ok(out.segments.into_iter().enumerate().map(|(i, s)| Segment::new(i, s.start, s.end, s.text)).collect())
    }
}

// --- Translator ----------------------------------------------------------------

pub struct PyTranslator(PyProcess);

#[derive(Deserialize)]
struct TranslateOut {
    texts: Vec<String>,
}

impl PyTranslator {
    pub fn spawn(env: &PythonEnv, spec: PySpec) -> Result<Self> {
        Ok(Self(PyProcess::spawn(env, spec)?))
    }
}
impl Engine for PyTranslator {
    fn info(&self) -> EngineInfo {
        info_of(&self.0)
    }
}
impl Translator for PyTranslator {
    fn translate(&self, texts: &[&str], from: &Lang, to: &Lang) -> Result<Vec<String>> {
        let out: TranslateOut = self.0.call("translate", json!({ "texts": texts, "from": from.0, "to": to.0 }))?;
        Ok(out.texts)
    }
}

// --- Synthesizer ---------------------------------------------------------------

pub struct PySynthesizer {
    proc: PyProcess,
    tmp: PathBuf,
    counter: AtomicU64,
}

#[derive(Deserialize)]
struct SynthOut {
    path: PathBuf,
}

impl PySynthesizer {
    pub fn spawn(env: &PythonEnv, spec: PySpec, tmp_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(tmp_dir)?;
        Ok(Self { proc: PyProcess::spawn(env, spec)?, tmp: tmp_dir.to_path_buf(), counter: AtomicU64::new(0) })
    }
}
impl Engine for PySynthesizer {
    fn info(&self) -> EngineInfo {
        info_of(&self.proc)
    }
}
impl Synthesizer for PySynthesizer {
    fn synthesize(&self, text: &str, lang: &Lang, speaker_ref: Option<&Path>, speed: f32) -> Result<AudioBuffer> {
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        let out_path = self.tmp.join(format!("py_tts_{}_{n}.wav", std::process::id()));
        let out: SynthOut = self.proc.call(
            "synthesize",
            json!({
                "text": text,
                "lang": lang.0,
                "out_path": abs(&out_path),
                "speaker_ref": speaker_ref.map(abs),
                "speed": speed,
            }),
        )?;
        let audio = AudioBuffer::read_wav(&out.path)?;
        let _ = std::fs::remove_file(&out.path);
        Ok(audio)
    }
    fn supports_voice_cloning(&self) -> bool {
        self.proc.has_capability("clone")
    }
    fn supports_speed(&self) -> bool {
        self.proc.has_capability("speed")
    }
}

// --- Separator -----------------------------------------------------------------

pub struct PySeparator(PyProcess);

#[derive(Deserialize)]
struct SeparateOut {
    voice: PathBuf,
    background: PathBuf,
}

impl PySeparator {
    pub fn spawn(env: &PythonEnv, spec: PySpec) -> Result<Self> {
        Ok(Self(PyProcess::spawn(env, spec)?))
    }
}
impl Engine for PySeparator {
    fn info(&self) -> EngineInfo {
        info_of(&self.0)
    }
}
impl Separator for PySeparator {
    fn separate(&self, audio: &Path, out_dir: &Path) -> Result<(PathBuf, PathBuf)> {
        std::fs::create_dir_all(out_dir)?;
        let out: SeparateOut = self.0.call("separate", json!({ "audio": abs(audio), "out_dir": abs(out_dir) }))?;
        Ok((out.voice, out.background))
    }
}

// --- Diarizer ------------------------------------------------------------------

pub struct PyDiarizer {
    proc: PyProcess,
    speakers: RwLock<Vec<Speaker>>,
    _guard: Mutex<()>,
}

#[derive(Deserialize)]
struct DiarSeg {
    index: usize,
    start: f64,
    end: f64,
    text: String,
    speaker: Option<String>,
    #[serde(default)]
    overlapped: bool,
}
#[derive(Deserialize)]
struct DiarizeOut {
    segments: Vec<DiarSeg>,
    #[serde(default)]
    speakers: Vec<String>,
}

impl PyDiarizer {
    pub fn spawn(env: &PythonEnv, spec: PySpec) -> Result<Self> {
        Ok(Self { proc: PyProcess::spawn(env, spec)?, speakers: RwLock::new(vec![]), _guard: Mutex::new(()) })
    }
}
impl Engine for PyDiarizer {
    fn info(&self) -> EngineInfo {
        info_of(&self.proc)
    }
}
impl Diarizer for PyDiarizer {
    fn diarize(&self, audio: &Path, segments: Vec<Segment>) -> Result<Vec<Segment>> {
        let input: Vec<_> = segments
            .iter()
            .map(|s| json!({ "index": s.index, "start": s.start, "end": s.end, "text": s.source_text }))
            .collect();
        let out: DiarizeOut = self.proc.call("diarize", json!({ "audio": abs(audio), "segments": input }))?;
        if let Ok(mut w) = self.speakers.write() {
            *w = out.speakers.into_iter().map(Speaker).collect();
        }
        let mut result: Vec<Segment> = out
            .segments
            .into_iter()
            .map(|d| {
                let mut s = Segment::new(d.index, d.start, d.end, d.text);
                s.speaker = d.speaker.map(Speaker);
                s.overlapped = d.overlapped;
                s
            })
            .collect();
        result.sort_by_key(|s| s.index);
        Ok(result)
    }
    fn speakers(&self) -> Vec<Speaker> {
        self.speakers.read().map(|r| r.clone()).unwrap_or_default()
    }
}

fn abs(p: &Path) -> String {
    std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf()).to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use dub_core::{DubJob, EngineSet, Pipeline, PipelineConfig};
    use std::sync::Arc;

    fn env() -> PythonEnv {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        PythonEnv::discover(Some(&root)).expect("python")
    }

    #[test]
    fn mock_engine_roundtrip() {
        let env = env();
        let t = PyTranscriber::spawn(&env, PySpec::new("mock")).unwrap();
        assert_eq!(t.info().name, "mock");
        let dir = std::env::temp_dir().join(format!("dub-py-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let wav = dir.join("in.wav");
        AudioBuffer::silence(16000, 5.0).write_wav(&wav).unwrap();
        let segs = t.transcribe(&wav, Some(&Lang::en())).unwrap();
        assert_eq!(segs.len(), 3);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn full_pipeline_with_mock_processes() {
        let env = env();
        let dir = std::env::temp_dir().join(format!("dub-pyfull-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let wav = dir.join("in.wav");
        AudioBuffer::silence(16000, 6.0).write_wav(&wav).unwrap();

        let engines = EngineSet {
            transcriber: Arc::new(PyTranscriber::spawn(&env, PySpec::new("mock")).unwrap()),
            translator: Arc::new(PyTranslator::spawn(&env, PySpec::new("mock")).unwrap()),
            synthesizer: Arc::new(PySynthesizer::spawn(&env, PySpec::new("mock"), &dir.join("tmp")).unwrap()),
            separator: Some(Arc::new(PySeparator::spawn(&env, PySpec::new("mock")).unwrap())),
            diarizer: Some(Arc::new(PyDiarizer::spawn(&env, PySpec::new("mock")).unwrap())),
        };
        let cfg = PipelineConfig { sample_rate: 22050, ..Default::default() };
        let out = Pipeline::new(engines, cfg).run(&DubJob { audio: wav, work_dir: dir.clone() }, None).unwrap();
        assert_eq!(out.segments.len(), 3);
        assert!(out.segments.iter().all(|s| s.speaker.is_some()));
        assert!(out.background_wav.is_some());
        assert!(out.mixed_wav.exists());
        let _ = std::fs::remove_dir_all(dir);
    }
}
