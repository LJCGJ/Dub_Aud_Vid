use serde::{Deserialize, Serialize};
use std::fmt;

/// Código de idioma (ISO 639-1, com região opcional). Ex.: `en`, `pt-BR`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Lang(pub String);

impl Lang {
    pub fn en() -> Self {
        Lang("en".into())
    }
    pub fn pt_br() -> Self {
        Lang("pt-BR".into())
    }
    /// Apenas a parte do idioma, sem região (`pt-BR` → `pt`).
    pub fn base(&self) -> &str {
        self.0.split(['-', '_']).next().unwrap_or(&self.0)
    }
}

impl fmt::Display for Lang {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for Lang {
    fn from(s: &str) -> Self {
        Lang(s.to_string())
    }
}

/// Identificador de falante produzido pela diarização.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Speaker(pub String);

impl Speaker {
    pub fn unknown() -> Self {
        Speaker("SPEAKER_00".into())
    }
}

/// Unidade de trabalho do pipeline: um trecho de fala com tempo de início/fim
/// no áudio original. Nasce na transcrição e vai sendo enriquecido a cada etapa.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    /// Índice sequencial no vídeo (ordem de reprodução).
    pub index: usize,
    /// Início em segundos no áudio original.
    pub start: f64,
    /// Fim em segundos no áudio original.
    pub end: f64,
    /// Falante (preenchido pela diarização; `None` se desligada).
    pub speaker: Option<Speaker>,
    /// Trecho tem sobreposição de falas detectada.
    pub overlapped: bool,
    /// Texto no idioma original.
    pub source_text: String,
    /// Texto traduzido (preenchido pelo `Translator`).
    pub target_text: Option<String>,
    /// Caminho do WAV sintetizado para este segmento (preenchido pelo `Synthesizer`).
    pub dubbed_wav: Option<std::path::PathBuf>,
    /// Fator de velocidade aplicado para caber na janela (1.0 = sem ajuste).
    pub speed_factor: f32,
}

impl Segment {
    pub fn new(index: usize, start: f64, end: f64, source_text: impl Into<String>) -> Self {
        Self {
            index,
            start,
            end,
            speaker: None,
            overlapped: false,
            source_text: source_text.into(),
            target_text: None,
            dubbed_wav: None,
            speed_factor: 1.0,
        }
    }

    /// Duração disponível para a dublagem, em segundos.
    pub fn duration(&self) -> f64 {
        (self.end - self.start).max(0.0)
    }
}
