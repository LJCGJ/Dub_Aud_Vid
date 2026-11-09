//! Catálogo unificado de vozes (Piper + Kokoro) e o roteador que escolhe a voz
//! pelo gênero estimado de cada trecho.

use dub_core::engines::{Engine, EngineInfo, Synthesizer};
use dub_core::{AudioBuffer, Gender, Lang, Result, Segment};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

/// Nome especial: escolhe masculina/feminina por trecho.
pub const AUTO_VOICE: &str = "auto";
pub const DEFAULT_MALE_VOICE: &str = "pt_BR-faber-medium";
pub const DEFAULT_FEMALE_VOICE: &str = "pf_dora";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceInfo {
    /// Nome a passar em `--voice`.
    pub name: String,
    /// `auto`, `piper` ou `kokoro`.
    pub engine: String,
    /// `masculina`, `feminina` ou `automática`.
    pub gender: String,
    pub description: String,
    pub installed: bool,
    /// Tamanho do download se não estiver instalada (MB).
    pub download_mb: u32,
}

/// Todas as vozes conhecidas, com a automática primeiro.
pub fn list() -> Vec<VoiceInfo> {
    let mut out = vec![VoiceInfo {
        name: AUTO_VOICE.into(),
        engine: "auto".into(),
        gender: "automática".into(),
        description: format!(
            "Detecta se cada voz original é masculina ou feminina e usa {DEFAULT_MALE_VOICE} / {DEFAULT_FEMALE_VOICE}"
        ),
        installed: true,
        download_mb: 0,
    }];
    let kokoro_installed = kokoro_installed();
    #[cfg(feature = "native")]
    for (name, gender, desc) in dub_native::kokoro::PT_VOICES {
        out.push(VoiceInfo {
            name: (*name).into(),
            engine: "kokoro".into(),
            gender: (*gender).into(),
            description: (*desc).into(),
            installed: kokoro_installed,
            download_mb: if kokoro_installed { 0 } else { 197 },
        });
    }
    let _ = kokoro_installed;
    for (entry, status) in dub_models::summary()
        .into_iter()
        .find(|(k, _)| *k == dub_core::paths::ModelKind::Voice)
        .map(|(_, v)| v)
        .unwrap_or_default()
    {
        let installed = matches!(status, dub_models::InstallStatus::Installed(_));
        let gender = if entry.description.to_ascii_lowercase().contains("feminina") { "feminina" } else { "masculina" };
        out.push(VoiceInfo {
            name: entry.name.clone(),
            engine: "piper".into(),
            gender: gender.into(),
            description: format!("{} (Piper)", entry.description),
            installed,
            download_mb: if installed { 0 } else { entry.total_mb() },
        });
    }
    out
}

pub fn kokoro_installed() -> bool {
    #[cfg(feature = "native")]
    {
        dub_native::KokoroSynthesizer::model_available()
    }
    #[cfg(not(feature = "native"))]
    {
        false
    }
}

/// Entrada do catálogo necessária para baixar uma voz que falta.
pub fn catalog_entry_for(voice: &str) -> Option<dub_models::ModelEntry> {
    if is_kokoro_voice(voice) {
        return dub_models::find("kokoro-v1.0");
    }
    dub_models::find(voice)
}

pub fn is_kokoro_voice(name: &str) -> bool {
    #[cfg(feature = "native")]
    {
        dub_native::kokoro::is_kokoro_voice(name)
    }
    #[cfg(not(feature = "native"))]
    {
        let _ = name;
        false
    }
}

/// Tom típico (Hz) de cada voz conhecida, medido com o mesmo estimador (YIN)
/// usado nos trechos originais. Serve para escolher a voz mais próxima do
/// falante e para o deslocamento fino de tom ficar pequeno.
pub fn typical_f0(voice: &str) -> f32 {
    match voice {
        "pf_dora" => 188.0,
        "pm_alex" => 130.0,
        "pm_santa" => 135.0,
        "pt_BR-faber-medium" => 188.0,
        "pt_BR-edresson-low" => 120.0,
        "pt_BR-cadu-medium" => 125.0,
        "pt_BR-jeff-medium" => 125.0,
        v if is_kokoro_voice(v) && v.as_bytes()[1] == b'f' => 200.0,
        _ => 130.0,
    }
}

pub fn gender_of(voice: &str) -> Gender {
    if voice == "pf_dora" || (is_kokoro_voice(voice) && voice.as_bytes()[1] == b'f') {
        Gender::Female
    } else {
        Gender::Male
    }
}

/// Como uma voz candidata é sintetizada.
pub enum VoiceBackend {
    /// Um sintetizador dedicado (Piper: um modelo por voz).
    Own(Arc<dyn Synthesizer>),
    /// Uma voz do Kokoro compartilhando o mesmo modelo carregado.
    #[cfg(feature = "native")]
    Kokoro(Arc<dub_native::KokoroSynthesizer>),
}

pub struct VoiceCandidate {
    pub name: String,
    pub gender: Gender,
    pub f0: f32,
    pub backend: VoiceBackend,
}

impl VoiceCandidate {
    fn synthesize(&self, seg: &Segment, text: &str, lang: &Lang, speaker_ref: Option<&Path>, speed: f32) -> Result<AudioBuffer> {
        match &self.backend {
            VoiceBackend::Own(s) => s.synthesize_segment(seg, text, lang, speaker_ref, speed),
            #[cfg(feature = "native")]
            VoiceBackend::Kokoro(k) => k.synthesize_with_voice(&self.name, text, lang, speed),
        }
    }
    fn device(&self) -> String {
        match &self.backend {
            VoiceBackend::Own(s) => s.info().device,
            #[cfg(feature = "native")]
            VoiceBackend::Kokoro(k) => k.info().device,
        }
    }
}

/// Escolhe, por trecho, a voz candidata do mesmo gênero com o tom mais
/// próximo do falante original — e o pipeline ainda afina o resto (±4
/// semitons). Resultado: cada falante ganha uma voz, como no YouTube.
pub struct VoiceRouter {
    pub candidates: Vec<VoiceCandidate>,
}

impl VoiceRouter {
    fn pick(&self, gender: Option<Gender>, f0: Option<f32>) -> &VoiceCandidate {
        let same: Vec<&VoiceCandidate> = match gender {
            Some(g) => self.candidates.iter().filter(|c| c.gender == g).collect(),
            None => self.candidates.iter().collect(),
        };
        let pool = if same.is_empty() { self.candidates.iter().collect() } else { same };
        match f0 {
            Some(f) => pool
                .into_iter()
                .min_by(|a, b| (a.f0.ln() - f.ln()).abs().partial_cmp(&(b.f0.ln() - f.ln()).abs()).unwrap())
                .expect("há candidatas"),
            None => pool[0],
        }
    }
}

impl Engine for VoiceRouter {
    fn info(&self) -> EngineInfo {
        let names: Vec<String> = self
            .candidates
            .iter()
            .map(|c| format!("{} {}", c.name, if c.gender == Gender::Female { "♀" } else { "♂" }))
            .collect();
        let mut devices: Vec<String> = self.candidates.iter().map(|c| c.device()).collect();
        devices.sort();
        devices.dedup();
        EngineInfo { name: "auto".into(), model: names.join(" / "), device: devices.join(" / ") }
    }
}

impl Synthesizer for VoiceRouter {
    fn synthesize(&self, text: &str, lang: &Lang, speaker_ref: Option<&Path>, speed: f32) -> Result<AudioBuffer> {
        let seg = Segment::new(0, 0.0, 0.0, "");
        self.pick(None, None).synthesize(&seg, text, lang, speaker_ref, speed)
    }

    fn synthesize_segment(
        &self,
        segment: &Segment,
        text: &str,
        lang: &Lang,
        speaker_ref: Option<&Path>,
        speed: f32,
    ) -> Result<AudioBuffer> {
        self.pick(segment.gender, segment.f0_hz).synthesize(segment, text, lang, speaker_ref, speed)
    }

    fn supports_speed(&self) -> bool {
        true
    }
}
