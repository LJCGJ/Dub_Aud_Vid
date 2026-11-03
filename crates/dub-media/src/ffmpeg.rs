use dub_core::{DubError, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::debug;

/// Wrapper fino sobre os binários `ffmpeg` e `ffprobe`.
#[derive(Debug, Clone)]
pub struct Ffmpeg {
    ffmpeg: PathBuf,
    ffprobe: PathBuf,
}

#[derive(Debug, Clone)]
pub struct MediaInfo {
    pub duration: f64,
    pub has_video: bool,
    pub audio_streams: usize,
    pub sample_rate: Option<u32>,
}

#[derive(Deserialize)]
struct ProbeOut {
    streams: Vec<ProbeStream>,
    format: ProbeFormat,
}
#[derive(Deserialize)]
struct ProbeStream {
    codec_type: String,
    sample_rate: Option<String>,
}
#[derive(Deserialize)]
struct ProbeFormat {
    duration: Option<String>,
}

impl Ffmpeg {
    /// Localiza os binários no PATH (ou em `FFMPEG_DIR`, se definido).
    pub fn locate() -> Result<Self> {
        let find = |name: &str| -> Result<PathBuf> {
            if let Ok(dir) = std::env::var("FFMPEG_DIR") {
                let p = Path::new(&dir).join(format!("{name}{}", std::env::consts::EXE_SUFFIX));
                if p.exists() {
                    return Ok(p);
                }
            }
            which::which(name).map_err(|_| DubError::ToolMissing(name.to_string()))
        };
        Ok(Self { ffmpeg: find("ffmpeg")?, ffprobe: find("ffprobe")? })
    }

    pub fn probe(&self, input: &Path) -> Result<MediaInfo> {
        let out = Command::new(&self.ffprobe)
            .args(["-v", "error", "-print_format", "json", "-show_streams", "-show_format"])
            .arg(input)
            .output()?;
        if !out.status.success() {
            return Err(DubError::Media(String::from_utf8_lossy(&out.stderr).trim().to_string()));
        }
        let p: ProbeOut = serde_json::from_slice(&out.stdout)?;
        let audio: Vec<_> = p.streams.iter().filter(|s| s.codec_type == "audio").collect();
        Ok(MediaInfo {
            duration: p.format.duration.and_then(|d| d.parse().ok()).unwrap_or(0.0),
            has_video: p.streams.iter().any(|s| s.codec_type == "video"),
            audio_streams: audio.len(),
            sample_rate: audio.first().and_then(|s| s.sample_rate.as_ref()).and_then(|r| r.parse().ok()),
        })
    }

    /// Extrai o áudio como WAV mono PCM 16-bit no sample rate pedido
    /// (16 kHz é o que whisper/pyannote esperam).
    pub fn extract_audio(&self, input: &Path, output: &Path, sample_rate: u32) -> Result<()> {
        self.run(&[
            "-y", "-i", &input.to_string_lossy(),
            "-vn", "-ac", "1", "-ar", &sample_rate.to_string(),
            "-c:a", "pcm_s16le",
            &output.to_string_lossy(),
        ])
    }

    /// Converte qualquer WAV para outro sample rate/canais (mantém PCM 16-bit).
    pub fn resample(&self, input: &Path, output: &Path, sample_rate: u32, channels: u8) -> Result<()> {
        self.run(&[
            "-y", "-i", &input.to_string_lossy(),
            "-ac", &channels.to_string(), "-ar", &sample_rate.to_string(),
            "-c:a", "pcm_s16le",
            &output.to_string_lossy(),
        ])
    }

    /// Altera a velocidade preservando o tom (filtro `atempo`). O filtro aceita
    /// 0.5..100 por estágio; encadeamos se precisar de valores fora de 0.5..2.0.
    pub fn atempo(&self, input: &Path, output: &Path, speed: f32) -> Result<()> {
        let filter = atempo_chain(speed);
        self.run(&[
            "-y", "-i", &input.to_string_lossy(),
            "-filter:a", &filter,
            "-c:a", "pcm_s16le",
            &output.to_string_lossy(),
        ])
    }

    /// Junta a trilha dublada ao vídeo original. Mantém o áudio original como
    /// segunda faixa (`keep_original`) para o espectador poder alternar, como no
    /// YouTube. Vídeo é copiado sem recodificar.
    pub fn remux(&self, video: &Path, dubbed_audio: &Path, output: &Path, keep_original: bool, lang_tag: &str) -> Result<()> {
        let v = video.to_string_lossy().to_string();
        let a = dubbed_audio.to_string_lossy().to_string();
        let o = output.to_string_lossy().to_string();
        let mut args: Vec<String> = vec!["-y".into(), "-i".into(), v, "-i".into(), a];
        args.extend(["-map", "0:v:0", "-map", "1:a:0"].iter().map(|s| s.to_string()));
        if keep_original {
            args.extend(["-map", "0:a:0?"].iter().map(|s| s.to_string()));
        }
        args.extend(["-c:v", "copy", "-c:a", "aac", "-b:a", "192k"].iter().map(|s| s.to_string()));
        args.extend([
            "-metadata:s:a:0".to_string(), format!("language={}", lang_tag),
            "-metadata:s:a:0".to_string(), "title=Dublado".to_string(),
            "-disposition:a:0".to_string(), "default".to_string(),
        ]);
        if keep_original {
            args.extend(["-metadata:s:a:1".to_string(), "title=Original".to_string(), "-disposition:a:1".to_string(), "0".to_string()]);
        }
        args.extend(["-shortest".to_string(), o]);
        let refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
        self.run(&refs)
    }

    /// Embute o SRT como legenda (soft subtitle) num MP4/MKV já remuxado.
    pub fn add_subtitles(&self, video: &Path, srt: &Path, output: &Path, lang_tag: &str) -> Result<()> {
        let codec = if output.extension().map(|e| e == "mkv").unwrap_or(false) { "srt" } else { "mov_text" };
        self.run(&[
            "-y", "-i", &video.to_string_lossy(), "-i", &srt.to_string_lossy(),
            "-map", "0", "-map", "1", "-c", "copy", "-c:s", codec,
            "-metadata:s:s:0", &format!("language={lang_tag}"),
            &output.to_string_lossy(),
        ])
    }

    fn run(&self, args: &[&str]) -> Result<()> {
        debug!(?args, "ffmpeg");
        let out = Command::new(&self.ffmpeg).args(["-hide_banner", "-loglevel", "error"]).args(args).output()?;
        if out.status.success() {
            Ok(())
        } else {
            Err(DubError::Media(String::from_utf8_lossy(&out.stderr).trim().to_string()))
        }
    }
}

/// Monta a cadeia de `atempo` para qualquer fator (cada estágio em 0.5..2.0).
pub fn atempo_chain(speed: f32) -> String {
    let mut remaining = speed.max(0.01);
    let mut parts = Vec::new();
    while remaining > 2.0 {
        parts.push("atempo=2.0".to_string());
        remaining /= 2.0;
    }
    while remaining < 0.5 {
        parts.push("atempo=0.5".to_string());
        remaining /= 0.5;
    }
    parts.push(format!("atempo={remaining:.4}"));
    parts.join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atempo_chain_splits() {
        assert_eq!(atempo_chain(1.3), "atempo=1.3000");
        assert_eq!(atempo_chain(3.0), "atempo=2.0,atempo=1.5000");
        assert_eq!(atempo_chain(0.3), "atempo=0.5,atempo=0.6000");
    }
}
