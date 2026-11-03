//! Transcrição nativa com whisper.cpp (`whisper-rs`).
//!
//! Modelos são arquivos GGML (`ggml-medium.bin`, `ggml-large-v3-q5_0.bin`...).
//! Procura em `DUB_MODELS_DIR/whisper/`, depois em `~/.cache/dub/whisper/`.
//! Baixe de https://huggingface.co/ggerganov/whisper.cpp/tree/main .

use dub_core::engines::{Engine, EngineInfo, Transcriber};
use dub_core::{AudioBuffer, Device, DubError, Lang, Result, Segment};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tracing::{info, warn};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState};

pub struct WhisperTranscriber {
    ctx: WhisperContext,
    // WhisperState não é Sync; o pipeline só chama transcribe de uma thread,
    // mas o trait exige Sync — o mutex garante.
    state: Mutex<WhisperState>,
    model_name: String,
    device: Device,
    threads: usize,
}

impl WhisperTranscriber {
    /// `model` pode ser um caminho para o `.bin` ou um nome curto (`medium`,
    /// `large-v3`, `small-q5_1`...) que será resolvido nos diretórios de modelos.
    pub fn load(model: &str, device: Device, threads: usize) -> Result<Self> {
        let path = resolve_model(model)?;
        whisper_rs::install_logging_hooks();

        // Sem backend de GPU compilado, o whisper.cpp roda em CPU mesmo com use_gpu=true.
        let device = effective_device(device);
        let mut params = WhisperContextParameters::default();
        params.use_gpu(device.is_gpu());
        // flash attention acelera bastante em GPU; em CPU é neutro.
        params.flash_attn(device.is_gpu());

        info!(model = %path.display(), %device, "carregando whisper.cpp");
        let ctx = WhisperContext::new_with_params(&path.to_string_lossy(), params).map_err(|e| DubError::EngineUnavailable {
            engine: "whisper.cpp".into(),
            reason: format!("{e} ({})", path.display()),
        })?;
        let state = ctx.create_state().map_err(|e| DubError::EngineUnavailable { engine: "whisper.cpp".into(), reason: e.to_string() })?;

        let model_name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| model.to_string());
        Ok(Self { ctx, state: Mutex::new(state), model_name, device, threads: threads.max(1) })
    }

    /// Verdadeiro se existe algum modelo resolvível com esse nome.
    pub fn model_available(model: &str) -> bool {
        resolve_model(model).is_ok()
    }
}

impl Engine for WhisperTranscriber {
    fn info(&self) -> EngineInfo {
        EngineInfo { name: "whisper.cpp".into(), model: self.model_name.clone(), device: self.device.to_string() }
    }
}

impl Transcriber for WhisperTranscriber {
    fn transcribe(&self, audio: &Path, lang_hint: Option<&Lang>) -> Result<Vec<Segment>> {
        let buf = AudioBuffer::read_wav(audio)?;
        let samples = to_16k(&buf);

        let mut params = FullParams::new(SamplingStrategy::BeamSearch { beam_size: 5, patience: -1.0 });
        params.set_n_threads(self.threads as i32);
        params.set_translate(false);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_token_timestamps(true);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);
        // Sem contexto entre janelas: evita repetições em loop em vídeos longos.
        params.set_no_context(true);
        // Segmentos curtos (≈ uma frase) dão time-fit melhor do que blocos de 30 s.
        params.set_max_len(90);
        params.set_split_on_word(true);
        let lang_str = lang_hint.map(|l| l.base().to_string());
        params.set_language(lang_str.as_deref());

        let mut state = self.state.lock().map_err(|_| DubError::ChannelClosed("whisper state"))?;
        state.full(params, &samples).map_err(|e| DubError::EngineFailed {
            engine: "whisper.cpp".into(),
            stage: "transcribe",
            message: e.to_string(),
        })?;

        let n = state.full_n_segments().map_err(|e| wf(e.to_string()))?;
        let mut out = Vec::with_capacity(n as usize);
        for i in 0..n {
            let t0 = state.full_get_segment_t0(i).map_err(|e| wf(e.to_string()))? as f64 / 100.0;
            let t1 = state.full_get_segment_t1(i).map_err(|e| wf(e.to_string()))? as f64 / 100.0;
            let text = state.full_get_segment_text_lossy(i).map_err(|e| wf(e.to_string()))?;
            let text = text.trim().to_string();
            if text.is_empty() || is_noise_marker(&text) {
                continue;
            }
            out.push(Segment::new(out.len(), t0, t1, text));
        }
        let _ = &self.ctx;
        Ok(merge_tiny(out))
    }
}

/// Dispositivo que o whisper.cpp realmente vai usar, dado o backend compilado.
pub fn effective_device(requested: Device) -> Device {
    match requested {
        Device::Cuda if cfg!(feature = "whisper-cuda") => Device::Cuda,
        Device::Vulkan | Device::DirectMl if cfg!(feature = "whisper-vulkan") => Device::Vulkan,
        Device::Metal if cfg!(feature = "whisper-metal") => Device::Metal,
        _ => {
            if requested.is_gpu() {
                warn!(%requested, "whisper.cpp compilado sem esse backend de GPU; usando CPU (features: whisper-cuda / whisper-vulkan)");
            }
            Device::Cpu
        }
    }
}

fn wf(message: String) -> DubError {
    DubError::EngineFailed { engine: "whisper.cpp".into(), stage: "transcribe", message }
}

/// Whisper emite marcadores como "[MUSIC]", "(applause)", "♪" em trechos sem fala.
fn is_noise_marker(t: &str) -> bool {
    let s = t.trim();
    (s.starts_with('[') && s.ends_with(']')) || (s.starts_with('(') && s.ends_with(')')) || s.chars().all(|c| !c.is_alphanumeric())
}

/// Junta segmentos muito curtos (< 0,6 s) ao anterior — poucas palavras isoladas
/// viram dublagens picotadas.
fn merge_tiny(segs: Vec<Segment>) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::with_capacity(segs.len());
    for s in segs {
        if let Some(last) = out.last_mut() {
            if s.duration() < 0.6 && s.start - last.end < 0.3 {
                last.end = s.end;
                last.source_text.push(' ');
                last.source_text.push_str(&s.source_text);
                continue;
            }
        }
        out.push(s);
    }
    for (i, s) in out.iter_mut().enumerate() {
        s.index = i;
    }
    out
}

/// whisper.cpp exige 16 kHz mono. Reamostragem linear se vier diferente.
fn to_16k(a: &AudioBuffer) -> Vec<f32> {
    if a.sample_rate == 16_000 {
        return a.samples.clone();
    }
    warn!(sr = a.sample_rate, "áudio não está em 16 kHz; reamostrando para o whisper");
    let ratio = a.sample_rate as f64 / 16_000.0;
    let n = (a.samples.len() as f64 / ratio) as usize;
    (0..n)
        .map(|i| {
            let pos = i as f64 * ratio;
            let i0 = pos as usize;
            let frac = (pos - i0 as f64) as f32;
            let x0 = a.samples.get(i0).copied().unwrap_or(0.0);
            let x1 = a.samples.get(i0 + 1).copied().unwrap_or(x0);
            x0 + (x1 - x0) * frac
        })
        .collect()
}

/// Diretórios onde procurar modelos, em ordem.
pub fn model_dirs() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(d) = std::env::var_os("DUB_MODELS_DIR") {
        v.push(PathBuf::from(d).join("whisper"));
    }
    if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
        v.push(PathBuf::from(home).join(".cache").join("dub").join("whisper"));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            v.push(dir.join("models").join("whisper"));
        }
    }
    v
}

fn resolve_model(model: &str) -> Result<PathBuf> {
    let p = Path::new(model);
    if p.is_file() {
        return Ok(p.to_path_buf());
    }
    let candidates = [format!("ggml-{model}.bin"), format!("{model}.bin"), model.to_string()];
    for dir in model_dirs() {
        for c in &candidates {
            let f = dir.join(c);
            if f.is_file() {
                return Ok(f);
            }
        }
    }
    Err(DubError::EngineUnavailable {
        engine: "whisper.cpp".into(),
        reason: format!(
            "modelo '{model}' não encontrado em {}. Baixe ggml-{model}.bin de huggingface.co/ggerganov/whisper.cpp",
            model_dirs().iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join(", ")
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_markers() {
        assert!(is_noise_marker("[MUSIC]"));
        assert!(is_noise_marker("(applause)"));
        assert!(is_noise_marker("♪ ♪"));
        assert!(!is_noise_marker("Hello there"));
    }

    #[test]
    fn merges_tiny_segments() {
        let v = vec![Segment::new(0, 0.0, 2.0, "Hello"), Segment::new(1, 2.1, 2.4, "there"), Segment::new(2, 4.0, 6.0, "Bye")];
        let m = merge_tiny(v);
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].source_text, "Hello there");
        assert_eq!(m[1].index, 1);
    }

    #[test]
    fn missing_model_is_clear_error() {
        let e = resolve_model("definitely-not-a-model").unwrap_err().to_string();
        assert!(e.contains("não encontrado"));
    }
}
