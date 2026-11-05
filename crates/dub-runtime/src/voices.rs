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

/// Escolhe entre uma voz masculina e uma feminina pelo gênero do segmento.
pub struct GenderRouter {
    pub male: Arc<dyn Synthesizer>,
    pub female: Arc<dyn Synthesizer>,
    pub male_name: String,
    pub female_name: String,
}

impl Engine for GenderRouter {
    fn info(&self) -> EngineInfo {
        let m = self.male.info();
        let f = self.female.info();
        EngineInfo {
            name: "auto".into(),
            model: format!("{} ♂ / {} ♀", self.male_name, self.female_name),
            device: if m.device == f.device { m.device } else { format!("{} / {}", m.device, f.device) },
        }
    }
}

impl Synthesizer for GenderRouter {
    fn synthesize(&self, text: &str, lang: &Lang, speaker_ref: Option<&Path>, speed: f32) -> Result<AudioBuffer> {
        self.male.synthesize(text, lang, speaker_ref, speed)
    }

    fn synthesize_segment(
        &self,
        segment: &Segment,
        text: &str,
        lang: &Lang,
        speaker_ref: Option<&Path>,
        speed: f32,
    ) -> Result<AudioBuffer> {
        match segment.gender {
            Some(Gender::Female) => self.female.synthesize_segment(segment, text, lang, speaker_ref, speed),
            _ => self.male.synthesize_segment(segment, text, lang, speaker_ref, speed),
        }
    }

    fn supports_speed(&self) -> bool {
        self.male.supports_speed() && self.female.supports_speed()
    }
}
