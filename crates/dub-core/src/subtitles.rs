//! Legendas prontas (SRT / WebVTT) como fonte de transcrição.
//!
//! Para música e para vídeos com legenda oficial, a legenda do próprio vídeo
//! costuma ser muito melhor do que o whisper ouvindo canto em cima da batida.
//! [`SubtitleTranscriber`] lê o arquivo e devolve os mesmos `Segment`s que um
//! transcritor de áudio devolveria, então o resto do pipeline não muda.

use crate::engines::{Engine, EngineInfo, Transcriber};
use crate::error::{DubError, Result};
use crate::segment::{Lang, Segment};
use std::path::{Path, PathBuf};

pub struct SubtitleTranscriber {
    path: PathBuf,
}

impl SubtitleTranscriber {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Engine for SubtitleTranscriber {
    fn info(&self) -> EngineInfo {
        let model = self.path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
        EngineInfo { name: "legenda".into(), model, device: "arquivo".into() }
    }
}

impl Transcriber for SubtitleTranscriber {
    fn transcribe(&self, _audio: &Path, _lang_hint: Option<&Lang>) -> Result<Vec<Segment>> {
        let text = std::fs::read_to_string(&self.path)
            .map_err(|e| DubError::Media(format!("ler legenda {}: {e}", self.path.display())))?;
        let segs = parse_subtitles(&text);
        if segs.is_empty() {
            return Err(DubError::Media(format!("legenda vazia ou em formato desconhecido: {}", self.path.display())));
        }
        Ok(segs)
    }
}

/// Lê SRT ou WebVTT (inclusive o VTT "rolante" das legendas automáticas do
/// YouTube, em que cada linha aparece duas vezes com marcas por palavra).
pub fn parse_subtitles(text: &str) -> Vec<Segment> {
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut cues: Vec<(f64, f64, String, bool)> = Vec::new(); // (start, end, texto, tinha marcas por palavra)
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        let Some((start, end)) = parse_timing_line(line) else { continue };
        let mut raw_lines: Vec<&str> = Vec::new();
        while let Some(&next) = lines.peek() {
            if next.trim().is_empty() || parse_timing_line(next).is_some() {
                break;
            }
            raw_lines.push(lines.next().unwrap_or_default());
        }
        // No VTT rolante do YouTube cada cue traz a linha anterior (sem marcas)
        // e a nova (com marcas por palavra); só a nova interessa.
        let has_word_marks = |l: &&str| l.contains("<c") || l.contains("><");
        let tagged = raw_lines.iter().any(has_word_marks);
        let body: String =
            raw_lines.iter().filter(|l| !tagged || has_word_marks(l)).map(|l| strip_tags(l)).collect::<Vec<_>>().join(" ");
        let body = normalize_ws(&body);
        if body.is_empty() || end <= start {
            continue;
        }
        cues.push((start, end, body, tagged));
    }

    // Legenda automática do YouTube: só as entradas com marcas por palavra são
    // "de verdade"; as outras são a mesma linha repetida para rolar na tela.
    if cues.iter().any(|c| c.3) {
        cues.retain(|c| c.3);
    }

    // Remove repetições consecutivas e linhas que são prefixo da seguinte.
    let mut out: Vec<Segment> = Vec::with_capacity(cues.len());
    for (start, end, body, _) in cues {
        if let Some(last) = out.last_mut() {
            if last.source_text == body {
                last.end = last.end.max(end);
                continue;
            }
            if body.starts_with(&last.source_text) && start - last.start < 0.5 {
                last.source_text = body;
                last.end = last.end.max(end);
                continue;
            }
        }
        out.push(Segment::new(out.len(), start, end, body));
    }
    // Cues de duração mínima (o VTT rolante usa 0,01 s): estende até o próximo.
    for i in 0..out.len() {
        if out[i].duration() < 0.3 {
            let next = out.get(i + 1).map(|n| n.start).unwrap_or(out[i].start + 2.0);
            out[i].end = (out[i].start + 2.0).min(next.max(out[i].start + 0.3));
        }
    }
    out
}

fn parse_timing_line(line: &str) -> Option<(f64, f64)> {
    let line = line.trim();
    let (a, rest) = line.split_once("-->")?;
    let b = rest.split_whitespace().next()?;
    Some((parse_time(a.trim())?, parse_time(b)?))
}

/// `HH:MM:SS,mmm`, `HH:MM:SS.mmm` ou `MM:SS.mmm`.
fn parse_time(s: &str) -> Option<f64> {
    let s = s.replace(',', ".");
    let parts: Vec<&str> = s.split(':').collect();
    let (h, m, sec) = match parts.len() {
        3 => (parts[0].parse::<f64>().ok()?, parts[1].parse::<f64>().ok()?, parts[2].parse::<f64>().ok()?),
        2 => (0.0, parts[0].parse::<f64>().ok()?, parts[1].parse::<f64>().ok()?),
        _ => return None,
    };
    Some(h * 3600.0 + m * 60.0 + sec)
}

fn strip_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
}

fn normalize_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// `true` se o texto parece um arquivo de legenda (SRT/VTT).
pub fn looks_like_subtitles(text: &str) -> bool {
    text.lines().take(50).any(|l| parse_timing_line(l).is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_srt() {
        let srt = "1\n00:00:00,000 --> 00:00:03,172\nHello there.\n\n2\n00:00:03,172 --> 00:00:05,920\nLet's go\non a trip.\n\n";
        let s = parse_subtitles(srt);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].source_text, "Hello there.");
        assert_eq!(s[1].source_text, "Let's go on a trip.");
        assert!((s[1].end - 5.92).abs() < 1e-6);
    }

    #[test]
    fn parses_rolling_youtube_vtt() {
        let vtt = "WEBVTT\nKind: captions\nLanguage: en\n\n\
00:00:00.000 --> 00:00:02.000 align:start position:0%\nGreetings<00:00:00.500><c> loved</c><00:00:01.000><c> ones</c>\n\n\
00:00:02.000 --> 00:00:02.010 align:start position:0%\nGreetings loved ones\n\n\
00:00:02.010 --> 00:00:04.000 align:start position:0%\nGreetings loved ones\nlet's<00:00:02.500><c> take</c><00:00:03.000><c> a</c><00:00:03.500><c> journey</c>\n\n";
        let s = parse_subtitles(vtt);
        assert_eq!(s.len(), 2, "{s:?}");
        assert_eq!(s[0].source_text, "Greetings loved ones");
        assert_eq!(s[1].source_text, "let's take a journey");
        assert!((s[1].start - 2.01).abs() < 1e-6);
    }

    #[test]
    fn detects_subtitle_text() {
        assert!(looks_like_subtitles("1\n00:00:01,000 --> 00:00:02,000\nx\n"));
        assert!(!looks_like_subtitles("just some prose\nwithout timing"));
    }
}
