//! Síntese Piper (VITS) nativa: espeak-ng → ids → ONNX Runtime → áudio.
//!
//! Um **pacote de voz** Piper é um par `nome.onnx` + `nome.onnx.json`. O JSON
//! traz sample rate, voz espeak, mapa de fonemas→ids e escalas de inferência.
//!
//! O ONNX Runtime é carregado dinamicamente (`load-dynamic`): procuramos a
//! biblioteca em `DUB_ORT_LIB`, ao lado do executável, e por fim no pacote
//! Python `onnxruntime` se houver. Trocar a DLL pela build DirectML/CUDA dá GPU.

use crate::phonemize::{phonemes_to_ids, phonemize};
use dub_core::engines::{Engine, EngineInfo, Synthesizer};
use dub_core::{AudioBuffer, Device, DubError, Lang, Result};
use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use tracing::{info, warn};

#[derive(Debug, Clone, Deserialize)]
pub struct VoiceConfig {
    pub audio: AudioCfg,
    pub espeak: EspeakCfg,
    #[serde(default)]
    pub inference: InferenceCfg,
    #[serde(default)]
    pub phoneme_type: Option<String>,
    #[serde(default)]
    pub phoneme_map: HashMap<String, Vec<String>>,
    pub phoneme_id_map: HashMap<String, Vec<i64>>,
    #[serde(default = "one")]
    pub num_speakers: u32,
    #[serde(default)]
    pub speaker_id_map: HashMap<String, i64>,
    #[serde(default)]
    pub language: Option<LanguageCfg>,
}
fn one() -> u32 {
    1
}
#[derive(Debug, Clone, Deserialize)]
pub struct AudioCfg {
    pub sample_rate: u32,
}
#[derive(Debug, Clone, Deserialize)]
pub struct EspeakCfg {
    pub voice: String,
}
#[derive(Debug, Clone, Deserialize)]
pub struct InferenceCfg {
    #[serde(default = "d_noise")]
    pub noise_scale: f32,
    #[serde(default = "d_len")]
    pub length_scale: f32,
    #[serde(default = "d_noise_w")]
    pub noise_w: f32,
}
fn d_noise() -> f32 {
    0.667
}
fn d_len() -> f32 {
    1.0
}
fn d_noise_w() -> f32 {
    0.8
}
impl Default for InferenceCfg {
    fn default() -> Self {
        Self { noise_scale: d_noise(), length_scale: d_len(), noise_w: d_noise_w() }
    }
}
#[derive(Debug, Clone, Deserialize)]
pub struct LanguageCfg {
    pub code: Option<String>,
}

/// Pacote de voz Piper resolvido no disco.
#[derive(Debug, Clone)]
pub struct VoicePack {
    pub name: String,
    pub onnx: PathBuf,
    pub config: VoiceConfig,
}

impl VoicePack {
    /// `spec` pode ser o caminho do `.onnx` ou um nome (`pt_BR-faber-medium`)
    /// procurado em `DUB_MODELS_DIR/voices`, `~/.cache/dub/voices` e `<exe>/voices`.
    pub fn resolve(spec: &str) -> Result<Self> {
        let onnx = {
            let p = Path::new(spec);
            if p.is_file() {
                p.to_path_buf()
            } else {
                let file = if spec.ends_with(".onnx") { spec.to_string() } else { format!("{spec}.onnx") };
                voice_dirs().into_iter().map(|d| d.join(&file)).find(|p| p.is_file()).ok_or_else(|| {
                    DubError::EngineUnavailable {
                        engine: "piper".into(),
                        reason: format!(
                            "pacote de voz '{spec}' não encontrado em {}. Baixe de huggingface.co/rhasspy/piper-voices",
                            voice_dirs().iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join(", ")
                        ),
                    }
                })?
            }
        };
        let json = PathBuf::from(format!("{}.json", onnx.display()));
        let cfg: VoiceConfig = serde_json::from_slice(&std::fs::read(&json).map_err(|e| DubError::EngineUnavailable {
            engine: "piper".into(),
            reason: format!("config {} ausente: {e}", json.display()),
        })?)?;
        if cfg.phoneme_type.as_deref().unwrap_or("espeak") != "espeak" {
            return Err(DubError::EngineUnavailable {
                engine: "piper".into(),
                reason: "só vozes com phoneme_type=espeak são suportadas".into(),
            });
        }
        let name = onnx.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| spec.into());
        Ok(Self { name, onnx, config: cfg })
    }

    pub fn lang_code(&self) -> Option<&str> {
        self.config.language.as_ref().and_then(|l| l.code.as_deref())
    }
}

pub fn voice_dirs() -> Vec<PathBuf> {
    dub_core::paths::model_dirs(dub_core::paths::ModelKind::Voice)
}

// --- ONNX Runtime -----------------------------------------------------------------

static ORT_ENV: OnceLock<std::result::Result<ort::environment::Environment, String>> = OnceLock::new();

/// Localiza a biblioteca do ONNX Runtime para carregamento dinâmico.
pub fn find_ort_lib() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("DUB_ORT_LIB").or_else(|| std::env::var_os("ORT_DYLIB_PATH")) {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    let names: &[&str] = if cfg!(windows) {
        &["onnxruntime.dll"]
    } else if cfg!(target_os = "macos") {
        &["libonnxruntime.dylib"]
    } else {
        &["libonnxruntime.so", "libonnxruntime.so.1"]
    };
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for n in names {
                let p = dir.join(n);
                if p.is_file() {
                    return Some(p);
                }
            }
        }
    }
    // Pacote Python onnxruntime (vem com piper-tts / faster-whisper).
    if let Ok(out) = std::process::Command::new(if cfg!(windows) { "python" } else { "python3" })
        .args(["-c", "import onnxruntime,os,glob;d=os.path.join(os.path.dirname(onnxruntime.__file__),'capi');print(next(iter(sorted(glob.glob(os.path.join(d,'*onnxruntime*.so*'))+glob.glob(os.path.join(d,'onnxruntime.dll'))+glob.glob(os.path.join(d,'*onnxruntime*.dylib')), key=len)),''))"])
        .output()
    {
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !s.is_empty() && Path::new(&s).is_file() {
            return Some(PathBuf::from(s));
        }
    }
    None
}

fn ort_env() -> Result<&'static ort::environment::Environment> {
    let r = ORT_ENV.get_or_init(|| {
        let lib = find_ort_lib().ok_or_else(|| {
            "biblioteca onnxruntime não encontrada (defina DUB_ORT_LIB ou coloque onnxruntime.dll ao lado do executável)"
                .to_string()
        })?;
        info!(lib = %lib.display(), "carregando ONNX Runtime");
        ort::init_from(&lib).map_err(|e| e.to_string())?.with_name("dub").build().map_err(|e| e.to_string())
    });
    r.as_ref().map_err(|e| DubError::EngineUnavailable { engine: "onnxruntime".into(), reason: e.clone() })
}

pub fn ort_available() -> bool {
    find_ort_lib().is_some()
}

// --- Synthesizer --------------------------------------------------------------------

pub struct PiperSynthesizer {
    pack: VoicePack,
    session: Mutex<Session>,
    device: Device,
    speaker_id: Option<i64>,
    /// Silêncio entre sentenças, em segundos (Piper usa 0,2).
    pub sentence_silence: f32,
}

impl PiperSynthesizer {
    pub fn load(voice: &str, device: Device, threads: usize) -> Result<Self> {
        let pack = VoicePack::resolve(voice)?;
        let env = ort_env()?;
        let device = effective_device(device);

        let mut builder = Session::builder(env).map_err(ort_err)?;
        let mut eps: Vec<ort::ep::ExecutionProviderDispatch> = Vec::new();
        match device {
            #[cfg(feature = "ort-cuda")]
            Device::Cuda => eps.push(ort::ep::CUDA::default().build()),
            #[cfg(feature = "ort-directml")]
            Device::DirectMl => eps.push(ort::ep::DirectML::default().build()),
            _ => {}
        }
        eps.push(ort::ep::CPU::default().build());
        builder = builder.with_execution_providers(eps).map_err(|e| ort_msg(e.message()))?;
        builder = builder.with_intra_threads(threads.max(1)).map_err(|e| ort_msg(e.message()))?;
        info!(voice = %pack.name, %device, "carregando voz Piper");
        let session = builder.commit_from_file(&pack.onnx).map_err(ort_err)?;

        let speaker_id = if pack.config.num_speakers > 1 { Some(0) } else { None };
        Ok(Self { pack, session: Mutex::new(session), device, speaker_id, sentence_silence: 0.2 })
    }

    pub fn with_speaker(mut self, id: i64) -> Self {
        if self.pack.config.num_speakers > 1 {
            self.speaker_id = Some(id);
        }
        self
    }

    pub fn voice_pack(&self) -> &VoicePack {
        &self.pack
    }

    fn run_sentence(&self, ids: Vec<i64>, length_scale: f32) -> Result<Vec<f32>> {
        let n = ids.len();
        let inf = &self.pack.config.inference;
        let input = Tensor::from_array(([1usize, n], ids)).map_err(ort_err)?;
        let lengths = Tensor::from_array(([1usize], vec![n as i64])).map_err(ort_err)?;
        let scales = Tensor::from_array(([3usize], vec![inf.noise_scale, length_scale, inf.noise_w])).map_err(ort_err)?;

        let mut session = self.session.lock().map_err(|_| DubError::ChannelClosed("piper session"))?;
        let outputs = if let Some(sid) = self.speaker_id {
            let sid_t = Tensor::from_array(([1usize], vec![sid])).map_err(ort_err)?;
            session
                .run(ort::inputs!["input" => input, "input_lengths" => lengths, "scales" => scales, "sid" => sid_t])
                .map_err(ort_err)?
        } else {
            session.run(ort::inputs!["input" => input, "input_lengths" => lengths, "scales" => scales]).map_err(ort_err)?
        };
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(ort_err)?;
        Ok(data.iter().map(|v| v.clamp(-1.0, 1.0)).collect())
    }
}

/// Dispositivo efetivo: só GPU se o execution provider correspondente foi compilado
/// (e a onnxruntime.dll carregada precisa ser a build com esse provider).
pub fn effective_device(requested: Device) -> Device {
    match requested {
        Device::Cuda if cfg!(feature = "ort-cuda") => Device::Cuda,
        // DirectML roda em qualquer GPU do Windows, NVIDIA incluída.
        Device::Cuda | Device::DirectMl | Device::Vulkan if cfg!(feature = "ort-directml") => Device::DirectMl,
        _ => {
            if requested.is_gpu() {
                info!(%requested, "ONNX Runtime sem execution provider para esse dispositivo; usando CPU (features: ort-cuda / ort-directml)");
            }
            Device::Cpu
        }
    }
}

fn ort_err(e: ort::Error) -> DubError {
    ort_msg(&e.to_string())
}

fn ort_msg(message: &str) -> DubError {
    DubError::EngineFailed { engine: "piper".into(), stage: "synthesize", message: message.to_string() }
}

impl Engine for PiperSynthesizer {
    fn info(&self) -> EngineInfo {
        EngineInfo { name: "piper".into(), model: self.pack.name.clone(), device: self.device.to_string() }
    }
}

impl Synthesizer for PiperSynthesizer {
    fn synthesize(&self, text: &str, lang: &Lang, _speaker_ref: Option<&Path>, speed: f32) -> Result<AudioBuffer> {
        let sr = self.pack.config.audio.sample_rate;
        if text.trim().is_empty() {
            return Ok(AudioBuffer::new(sr, vec![]));
        }
        if let Some(code) = self.pack.lang_code() {
            if code.split(['-', '_']).next() != Some(lang.base()) {
                warn!(voice_lang = code, requested = %lang, "voz Piper em idioma diferente do pedido");
            }
        }
        let sentences = phonemize(&self.pack.config.espeak.voice, text)?;
        // length_scale é o inverso da velocidade.
        let length_scale = self.pack.config.inference.length_scale / speed.clamp(0.5, 2.0);
        let gap = vec![0.0f32; (sr as f32 * self.sentence_silence) as usize];

        let mut out: Vec<f32> = Vec::new();
        let n = sentences.len();
        for (i, phonemes) in sentences.into_iter().enumerate() {
            let ids = phonemes_to_ids(&phonemes, &self.pack.config.phoneme_id_map, &self.pack.config.phoneme_map);
            if ids.len() <= 3 {
                continue;
            }
            out.extend(self.run_sentence(ids, length_scale)?);
            if i + 1 < n {
                out.extend_from_slice(&gap);
            }
        }
        Ok(AudioBuffer::new(sr, out))
    }

    fn supports_speed(&self) -> bool {
        true
    }
}
