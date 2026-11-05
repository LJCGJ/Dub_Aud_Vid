//! # dub-input-video
//!
//! Adaptador "vídeo": recebe um arquivo local ou URL do YouTube, prepara o
//! áudio para o pipeline e, no fim, devolve o vídeo com a trilha dublada
//! (mantendo a original como segunda faixa) e a legenda em PT-BR embutida.
//!
//! O adaptador "game" (GameDub) seguirá a mesma interface: `prepare()` produz
//! o `DubJob`, `finish()` leva o `DubOutput` de volta ao formato de origem.

use dub_core::pipeline::DubOutput;
use dub_core::{DubError, DubJob, Result};
use dub_media::{Ffmpeg, YtDlp};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoOptions {
    /// Manter a faixa de áudio original como segunda faixa.
    pub keep_original_audio: bool,
    /// Embutir legenda SRT no idioma de destino.
    pub embed_subtitles: bool,
    /// Tag de idioma ISO 639-2 para as faixas (ex.: `por`).
    pub lang_tag: String,
}

impl Default for VideoOptions {
    fn default() -> Self {
        Self { keep_original_audio: true, embed_subtitles: true, lang_tag: "por".into() }
    }
}

pub struct VideoInput {
    ff: Ffmpeg,
    /// Caminho do vídeo de origem (baixado ou local).
    pub video: PathBuf,
    pub work_dir: PathBuf,
    pub title: String,
}

impl VideoInput {
    /// Aceita caminho local ou URL. URLs são baixadas para `work_dir/source/`.
    pub fn open(source: &str, work_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(work_dir)?;
        let ff = Ffmpeg::locate()?;
        let (video, title) = if YtDlp::is_url(source) {
            let yt = YtDlp::locate()?;
            let title = yt.title(source).unwrap_or_else(|_| "video".into());
            let path = yt.download(source, &work_dir.join("source"))?;
            (path, title)
        } else {
            let p = PathBuf::from(source);
            if !p.exists() {
                return Err(DubError::Media(format!("arquivo não encontrado: {}", p.display())));
            }
            let title = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "video".into());
            (p, title)
        };
        let info = ff.probe(&video)?;
        if info.audio_streams == 0 {
            return Err(DubError::Media("o vídeo não tem faixa de áudio".into()));
        }
        info!(title, duration = info.duration, "vídeo aberto");
        Ok(Self { ff, video, work_dir: work_dir.to_path_buf(), title })
    }

    /// Troca o diretório de trabalho (útil quando o nome final só é conhecido
    /// depois do download).
    pub fn with_work_dir(mut self, dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        self.work_dir = dir.to_path_buf();
        Ok(self)
    }

    /// Extrai o áudio mono 16 kHz e monta o `DubJob`.
    pub fn prepare(&self) -> Result<DubJob> {
        let audio = self.work_dir.join("source_16k.wav");
        self.ff.extract_audio(&self.video, &audio, 16_000)?;
        // Versão estéreo 44,1 kHz para separação voz/instrumental e mixagem.
        let hq = self.work_dir.join("source_44k.wav");
        let audio_hq = match self.ff.extract_audio_stereo(&self.video, &hq, 44_100) {
            Ok(()) => Some(hq),
            Err(e) => {
                tracing::warn!("sem áudio em alta qualidade ({e}); separação usará o mono 16 kHz");
                None
            }
        };
        Ok(DubJob { audio, audio_hq, work_dir: self.work_dir.clone() })
    }

    /// Remuxa a trilha dublada no vídeo. Devolve o caminho do arquivo final.
    pub fn finish(&self, out: &DubOutput, output: &Path, opts: &VideoOptions) -> Result<PathBuf> {
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if opts.embed_subtitles {
            let tmp = self.work_dir.join(format!("remux_tmp.{}", ext_of(output)));
            self.ff.remux(&self.video, &out.mixed_wav, &tmp, opts.keep_original_audio, &opts.lang_tag)?;
            self.ff.add_subtitles(&tmp, &out.srt, output, &opts.lang_tag)?;
            let _ = std::fs::remove_file(tmp);
        } else {
            self.ff.remux(&self.video, &out.mixed_wav, output, opts.keep_original_audio, &opts.lang_tag)?;
        }
        info!(output = %output.display(), "vídeo dublado gerado");
        Ok(output.to_path_buf())
    }

    /// Nome de saída sugerido: `<título>.dub.pt-BR.mp4` ao lado do original.
    pub fn default_output(&self, target_lang: &str) -> PathBuf {
        let dir = if YtDlp::is_url(&self.video.to_string_lossy()) || self.video.starts_with(&self.work_dir) {
            std::env::current_dir().unwrap_or_else(|_| self.work_dir.clone())
        } else {
            self.video.parent().map(Path::to_path_buf).unwrap_or_else(|| self.work_dir.clone())
        };
        dir.join(format!("{}.dub.{}.mp4", sanitize(&self.title), target_lang))
    }
}

fn ext_of(p: &Path) -> String {
    p.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_else(|| "mp4".into())
}

fn sanitize(s: &str) -> String {
    let cleaned: String =
        s.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' || c == ' ' { c } else { '_' }).collect();
    let trimmed = cleaned.trim().replace(' ', "_");
    if trimmed.is_empty() {
        "video".into()
    } else {
        trimmed.chars().take(80).collect()
    }
}
