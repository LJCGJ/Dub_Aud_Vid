//! Síntese com o Kokoro (82M) em ONNX, no mesmo ONNX Runtime do Piper.
//!
//! É a única opção nativa de **voz feminina** em pt-BR (`pf_dora`); também traz
//! `pm_alex` e `pm_santa` (masculinas). Entradas do grafo: `tokens` (ids dos
//! fonemas IPA, com 0 nas pontas), `style` (vetor 256 do arquivo de vozes,
//! indexado pelo número de fonemas) e `speed`. Saída: áudio a 24 kHz.
//!
//! Fonemização: espeak-ng (`pt-br`), como o próprio Kokoro faz fora do inglês.

use crate::kokoro_vocab::KOKORO_VOCAB;
use crate::phonemize::phonemize;
use crate::piper::{ort_err, OrtEngine};
use dub_core::engines::{Engine, EngineInfo, Synthesizer};
use dub_core::paths::{self, ModelKind};
use dub_core::{AudioBuffer, Device, DubError, Lang, Result};
use ort::value::Tensor;
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

pub const SAMPLE_RATE: u32 = 24_000;
const STYLE_DIM: usize = 256;
const MAX_TOKENS: usize = 510;
const MODEL_FILE: &str = "kokoro-v1.0.fp16.onnx";
const VOICES_FILE: &str = "voices-v1.0.bin";

pub struct KokoroSynthesizer {
    engine: OrtEngine,
    /// Estilos por voz: 510 vetores de 256 (um por comprimento de frase). O
    /// modelo é um só; só o vetor de estilo muda de voz para voz.
    styles: HashMap<String, Vec<f32>>,
    /// Voz padrão (usada por `synthesize`).
    voice: String,
    vocab: HashMap<char, i64>,
    pub sentence_silence: f32,
}

/// Vozes do Kokoro em português e o gênero de cada uma.
pub const PT_VOICES: &[(&str, &str, &str)] = &[
    ("pf_dora", "feminina", "Voz feminina, natural, 24 kHz (Kokoro)"),
    ("pm_alex", "masculina", "Voz masculina, natural, 24 kHz (Kokoro)"),
    ("pm_santa", "masculina", "Voz masculina grave, 24 kHz (Kokoro)"),
];

/// Nomes do Kokoro têm a forma `<idioma><f|m>_<nome>` (`pf_dora`, `am_adam`);
/// os do Piper são `pt_BR-…`, então o `_` na 3ª posição não basta.
pub fn is_kokoro_voice(name: &str) -> bool {
    PT_VOICES.iter().any(|(n, _, _)| *n == name)
        || (name.is_ascii()
            && name.len() > 3
            && !name.contains('-')
            && name.as_bytes()[0].is_ascii_lowercase()
            && matches!(name.as_bytes()[1], b'f' | b'm')
            && name.as_bytes()[2] == b'_')
}

impl KokoroSynthesizer {
    pub fn load(voice: &str, device: Device, threads: usize) -> Result<Self> {
        Self::load_many(&[voice], device, threads)
    }

    /// Carrega o modelo uma vez com várias vozes; a primeira é a padrão.
    pub fn load_many(voices: &[&str], device: Device, threads: usize) -> Result<Self> {
        let dir = model_dir()?;
        let device = effective_device(device);
        info!(voices = ?voices, %device, "carregando Kokoro");
        let engine = OrtEngine::open("kokoro", &dir.join(MODEL_FILE), device, threads)?;
        let mut styles = HashMap::new();
        for v in voices {
            styles.insert((*v).to_string(), load_voice(&dir.join(VOICES_FILE), v)?);
        }
        let vocab = KOKORO_VOCAB.iter().copied().collect();
        Ok(Self { engine, styles, voice: voices[0].to_string(), vocab, sentence_silence: 0.2 })
    }

    pub fn voices(&self) -> Vec<String> {
        self.styles.keys().cloned().collect()
    }

    /// Sintetiza com uma voz específica já carregada.
    pub fn synthesize_with_voice(&self, voice: &str, text: &str, lang: &Lang, speed: f32) -> Result<AudioBuffer> {
        let style = self.styles.get(voice).ok_or_else(|| DubError::EngineUnavailable {
            engine: "kokoro".into(),
            reason: format!("voz '{voice}' não carregada (carregadas: {})", self.voices().join(", ")),
        })?;
        self.synthesize_style(style, text, lang, speed)
    }

    pub fn model_available() -> bool {
        model_dir().is_ok()
    }

    pub fn voice_name(&self) -> &str {
        &self.voice
    }

    fn tokens(&self, phonemes: &[char]) -> Vec<i64> {
        phonemes.iter().filter_map(|c| self.vocab.get(c).copied()).collect()
    }

    fn run_sentence(&self, style: &[f32], ids: &[i64], speed: f32) -> Result<Vec<f32>> {
        let n = ids.len().min(MAX_TOKENS);
        let mut tokens = Vec::with_capacity(n + 2);
        tokens.push(0);
        tokens.extend_from_slice(&ids[..n]);
        tokens.push(0);
        let style_row = style[n * STYLE_DIM..(n + 1) * STYLE_DIM].to_vec();
        self.engine.run(|session| {
            let tokens_t = Tensor::from_array(([1usize, tokens.len()], tokens.clone())).map_err(ort_err)?;
            let style_t = Tensor::from_array(([1usize, STYLE_DIM], style_row.clone())).map_err(ort_err)?;
            let speed_t = Tensor::from_array(([1usize], vec![speed])).map_err(ort_err)?;
            let outputs =
                session.run(ort::inputs!["tokens" => tokens_t, "style" => style_t, "speed" => speed_t]).map_err(ort_err)?;
            let (_shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(ort_err)?;
            Ok(data.iter().map(|v| v.clamp(-1.0, 1.0)).collect())
        })
    }
}

/// O grafo fp16 do Kokoro tem um ConvTranspose (`/encoder/F0.1/pool`) que o
/// DirectML rejeita em tempo de execução; CUDA funciona. Começar na CPU evita
/// a falha e o custo de recarregar.
fn effective_device(requested: Device) -> Device {
    match crate::piper::effective_device(requested) {
        Device::DirectMl => {
            info!("Kokoro: DirectML não executa este grafo; usando CPU");
            Device::Cpu
        }
        d => d,
    }
}

fn model_dir() -> Result<PathBuf> {
    paths::find(ModelKind::Kokoro, MODEL_FILE)
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .filter(|d| d.join(VOICES_FILE).exists())
        .ok_or_else(|| DubError::EngineUnavailable {
            engine: "kokoro".into(),
            reason: format!(
                "modelo Kokoro não encontrado em {:?} (rode: dub models download kokoro-v1.0)",
                paths::model_dirs(ModelKind::Kokoro)
            ),
        })
}

/// Lê `voices-v1.0.bin` (um `.npz` do NumPy: zip de arquivos `.npy`) e devolve
/// o estilo da voz pedida como 510×256 floats.
fn load_voice(path: &Path, voice: &str) -> Result<Vec<f32>> {
    let file = std::fs::File::open(path).map_err(|e| DubError::Audio(format!("abrir {}: {e}", path.display())))?;
    let mut zip = zip::ZipArchive::new(file).map_err(|e| DubError::Audio(format!("vozes Kokoro: {e}")))?;
    let name = format!("{voice}.npy");
    let available: Vec<String> =
        zip.file_names().filter(|n| n.starts_with('p')).map(|n| n.trim_end_matches(".npy").to_string()).collect();
    let mut entry = zip.by_name(&name).map_err(|_| DubError::EngineUnavailable {
        engine: "kokoro".into(),
        reason: format!("voz '{voice}' não existe no Kokoro; em português: {}", available.join(", ")),
    })?;
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes).map_err(|e| DubError::Audio(e.to_string()))?;
    parse_npy_f32(&bytes, MAX_TOKENS * STYLE_DIM)
}

/// `.npy` versão 1.x/2.x, little-endian float32, ordem C.
fn parse_npy_f32(bytes: &[u8], expected: usize) -> Result<Vec<f32>> {
    let bad = |m: &str| DubError::Audio(format!("vozes Kokoro: .npy inválido ({m})"));
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(bad("assinatura"));
    }
    let major = bytes[6];
    let (header_len, start) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12)
    };
    let header = std::str::from_utf8(&bytes[start..start + header_len]).map_err(|_| bad("cabeçalho"))?;
    if !header.contains("'<f4'") && !header.contains("\"<f4\"") {
        return Err(bad("esperado float32 little-endian"));
    }
    if header.contains("'fortran_order': True") {
        return Err(bad("ordem Fortran"));
    }
    let data = &bytes[start + header_len..];
    let n = data.len() / 4;
    if n != expected {
        warn!(n, expected, "vozes Kokoro com tamanho inesperado");
    }
    Ok(data.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

impl Engine for KokoroSynthesizer {
    fn info(&self) -> EngineInfo {
        EngineInfo { name: "kokoro".into(), model: self.voice.clone(), device: self.engine.device().to_string() }
    }
}

impl KokoroSynthesizer {
    fn synthesize_style(&self, style: &[f32], text: &str, lang: &Lang, speed: f32) -> Result<AudioBuffer> {
        if text.trim().is_empty() {
            return Ok(AudioBuffer::new(SAMPLE_RATE, vec![]));
        }
        let espeak_voice = match lang.base() {
            "pt" => "pt-br",
            "en" => "en-us",
            other => other,
        };
        let sentences = phonemize(espeak_voice, text)?;
        let gap = vec![0.0f32; (SAMPLE_RATE as f32 * self.sentence_silence) as usize];
        let mut out: Vec<f32> = Vec::new();
        let n = sentences.len();
        for (i, phonemes) in sentences.into_iter().enumerate() {
            let ids = self.tokens(&phonemes);
            if ids.len() <= 2 {
                continue;
            }
            out.extend(self.run_sentence(style, &ids, speed.clamp(0.5, 2.0))?);
            if i + 1 < n {
                out.extend_from_slice(&gap);
            }
        }
        Ok(AudioBuffer::new(SAMPLE_RATE, out))
    }
}

impl Synthesizer for KokoroSynthesizer {
    fn synthesize(&self, text: &str, lang: &Lang, _speaker_ref: Option<&Path>, speed: f32) -> Result<AudioBuffer> {
        self.synthesize_with_voice(&self.voice.clone(), text, lang, speed)
    }

    fn supports_speed(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npy_header_parsing() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }";
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&1.5f32.to_le_bytes());
        bytes.extend_from_slice(&(-2.0f32).to_le_bytes());
        assert_eq!(parse_npy_f32(&bytes, 2).unwrap(), vec![1.5, -2.0]);
    }

    #[test]
    fn voice_name_detection() {
        assert!(is_kokoro_voice("pf_dora"));
        assert!(is_kokoro_voice("am_adam"));
        assert!(!is_kokoro_voice("pt_BR-faber-medium"));
        assert!(!is_kokoro_voice("auto"));
    }

    #[test]
    fn vocab_has_core_phonemes() {
        let v: HashMap<char, i64> = KOKORO_VOCAB.iter().copied().collect();
        for c in ['a', 'ˈ', 'ɐ', 'ʃ', ' ', '.', 'ʒ'] {
            assert!(v.contains_key(&c), "{c}");
        }
    }
}
