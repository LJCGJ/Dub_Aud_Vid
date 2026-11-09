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

/// Gênero da voz original, estimado pelo tom (f0) do trecho isolado.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Gender {
    Male,
    Female,
}

impl Gender {
    /// Classifica pela frequência fundamental mediana: vozes masculinas
    /// costumam ficar em 85–155 Hz e femininas em 165–255 Hz.
    pub fn from_f0(f0_hz: f32) -> Gender {
        if f0_hz < 165.0 {
            Gender::Male
        } else {
            Gender::Female
        }
    }
    pub fn label_pt(self) -> &'static str {
        match self {
            Gender::Male => "masculina",
            Gender::Female => "feminina",
        }
    }
}

/// Natureza do trecho: fala comum ou canto. Canto não vira TTS — a voz
/// original fica, e a tradução vai só para a legenda.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SegmentKind {
    #[default]
    Speech,
    Singing,
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
    /// Gênero estimado da voz original (para escolher a voz da dublagem).
    #[serde(default)]
    pub gender: Option<Gender>,
    /// Tom mediano (Hz) da voz original neste trecho (ou do falante).
    #[serde(default)]
    pub f0_hz: Option<f32>,
    /// "Brilho" da voz original: fração da energia acima de ~2 kHz (0..1).
    #[serde(default)]
    pub brightness: Option<f32>,
    /// Trecho tem sobreposição de falas detectada.
    pub overlapped: bool,
    /// Fala ou canto (ver [`SegmentKind`]).
    #[serde(default)]
    pub kind: SegmentKind,
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
            gender: None,
            f0_hz: None,
            brightness: None,
            overlapped: false,
            kind: SegmentKind::Speech,
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

    pub fn is_singing(&self) -> bool {
        self.kind == SegmentKind::Singing
    }
}

/// Divide trechos longos (> `max_secs`) em pedaços menores, repartindo o tempo
/// proporcionalmente aos caracteres. Tenta, nesta ordem: pontuação final
/// (`. ! ?`), vírgulas/ponto-e-vírgula e, por fim, contagem de palavras. Letra
/// de música e legenda automática raramente têm pontuação — sem isto, um bloco
/// de 20 s vira uma dublagem única impossível de encaixar.
pub fn split_long(segs: Vec<Segment>, max_secs: f64) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::with_capacity(segs.len());
    for s in segs {
        if s.duration() <= max_secs {
            out.push(s);
            continue;
        }
        let parts = split_text(&s.source_text, s.duration(), max_secs);
        if parts.len() < 2 {
            out.push(s);
            continue;
        }
        let total_chars: usize = parts.iter().map(|t| t.chars().count()).sum::<usize>().max(1);
        let mut t = s.start;
        for (i, part) in parts.iter().enumerate() {
            let frac = part.chars().count() as f64 / total_chars as f64;
            let end = if i + 1 == parts.len() { s.end } else { t + s.duration() * frac };
            let mut piece = s.clone();
            piece.start = t;
            piece.end = end;
            piece.source_text = part.clone();
            piece.target_text = None;
            out.push(piece);
            t = end;
        }
    }
    for (i, s) in out.iter_mut().enumerate() {
        s.index = i;
    }
    out
}

fn split_text(text: &str, duration: f64, max_secs: f64) -> Vec<String> {
    // Folga de 20%: as palavras não se repartem em pedaços exatamente iguais.
    let pieces_needed = (duration / (max_secs * 0.8)).ceil().max(2.0) as usize;
    // 1. Frases.
    let by_sentence = split_on(text, |c| matches!(c, '.' | '!' | '?'), 12);
    if by_sentence.len() >= 2 && longest_share(&by_sentence) * duration <= max_secs * 1.25 {
        return by_sentence;
    }
    // 2. Vírgulas e afins (dentro de cada frase).
    let by_clause: Vec<String> =
        by_sentence.iter().flat_map(|s| split_on(s, |c| matches!(c, ',' | ';' | ':' | '—'), 10)).collect();
    if by_clause.len() >= 2 && longest_share(&by_clause) * duration <= max_secs * 1.25 {
        return by_clause;
    }
    // 3. Palavras, em blocos de tamanho parecido.
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 4 {
        return vec![text.to_string()];
    }
    let n = pieces_needed.min(words.len() / 2).max(2);
    let per = (words.len() as f64 / n as f64).ceil() as usize;
    words.chunks(per.max(1)).map(|w| w.join(" ")).collect()
}

/// Corta o texto depois de cada caractere aceito por `is_sep` (seguido de
/// espaço ou fim), desde que o pedaço tenha ao menos `min_chars` caracteres.
fn split_on(text: &str, is_sep: impl Fn(char) -> bool, min_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = text.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        cur.push(c);
        let at_end = is_sep(c) && chars.get(i + 1).map(|n| n.is_whitespace()).unwrap_or(true);
        if at_end && cur.trim().chars().count() >= min_chars {
            out.push(cur.trim().to_string());
            cur.clear();
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

fn longest_share(parts: &[String]) -> f64 {
    let total: usize = parts.iter().map(|p| p.chars().count()).sum::<usize>().max(1);
    parts.iter().map(|p| p.chars().count()).max().unwrap_or(0) as f64 / total as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_long_uses_sentences_first() {
        let s = Segment::new(0, 0.0, 20.0, "First sentence is here. Second one follows it! Third one ends?");
        let out = split_long(vec![s], 8.0);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].start, 0.0);
        assert_eq!(out[2].end, 20.0);
        assert!(out[1].start > out[0].start && out[1].end > out[1].start);
        assert_eq!(out.iter().map(|s| s.index).collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    #[test]
    fn split_long_falls_back_to_commas_and_words() {
        // Letra sem pontuação final, com vírgulas.
        let s = Segment::new(0, 0.0, 20.6, "we drove along the coast, the radio was loud, nobody wanted to go home");
        let out = split_long(vec![s], 8.0);
        assert_eq!(out.len(), 3);
        // Sem pontuação nenhuma: quebra por palavras.
        let s = Segment::new(0, 0.0, 16.0, "one two three four five six seven eight nine ten eleven twelve");
        let out = split_long(vec![s], 8.0);
        assert!(out.len() >= 2, "{out:?}");
        assert!(out.iter().all(|p| p.duration() <= 8.5));
    }

    #[test]
    fn split_long_keeps_short_segments() {
        let out = split_long(vec![Segment::new(0, 0.0, 5.0, "short one"), Segment::new(1, 5.0, 7.0, "and two")], 8.0);
        assert_eq!(out.len(), 2);
    }
}
