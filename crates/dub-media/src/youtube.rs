use dub_core::{DubError, Result};
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::info;

/// Download de vídeos via `yt-dlp` (processo externo).
#[derive(Debug, Clone)]
pub struct YtDlp {
    bin: PathBuf,
}

impl YtDlp {
    pub fn locate() -> Result<Self> {
        // Aceita `yt-dlp` no PATH ou como módulo Python (`python -m yt_dlp`).
        if let Ok(p) = which::which("yt-dlp") {
            return Ok(Self { bin: p });
        }
        Err(DubError::ToolMissing("yt-dlp".into()))
    }

    pub fn is_url(s: &str) -> bool {
        s.starts_with("http://") || s.starts_with("https://")
    }

    /// Baixa o melhor MP4 (vídeo+áudio) para `out_dir`. Devolve o caminho do arquivo.
    pub fn download(&self, url: &str, out_dir: &Path) -> Result<PathBuf> {
        std::fs::create_dir_all(out_dir)?;
        let template = out_dir.join("%(id)s.%(ext)s");
        info!(url, "baixando vídeo");
        let out = Command::new(&self.bin)
            .args([
                "--no-playlist",
                "-f", "bv*[ext=mp4][height<=1080]+ba[ext=m4a]/b[ext=mp4]/b",
                "--merge-output-format", "mp4",
                "--print", "after_move:filepath",
                "--no-simulate", "--quiet", "--no-warnings",
                "-o", &template.to_string_lossy(),
                url,
            ])
            .output()?;
        if !out.status.success() {
            return Err(DubError::Media(format!("yt-dlp: {}", String::from_utf8_lossy(&out.stderr).trim())));
        }
        let path = String::from_utf8_lossy(&out.stdout).lines().last().unwrap_or("").trim().to_string();
        if path.is_empty() {
            return Err(DubError::Media("yt-dlp não informou o arquivo baixado".into()));
        }
        Ok(PathBuf::from(path))
    }

    /// Título do vídeo (para nomear a saída).
    pub fn title(&self, url: &str) -> Result<String> {
        let out = Command::new(&self.bin).args(["--no-playlist", "--print", "title", "--quiet", "--no-warnings", url]).output()?;
        if !out.status.success() {
            return Err(DubError::Media(String::from_utf8_lossy(&out.stderr).trim().to_string()));
        }
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    }
}
