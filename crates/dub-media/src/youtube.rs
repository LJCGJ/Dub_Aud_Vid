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
        let name = format!("yt-dlp{}", std::env::consts::EXE_SUFFIX);
        if let Ok(dir) = std::env::var("DUB_TOOLS_DIR") {
            let p = Path::new(&dir).join(&name);
            if p.exists() {
                return Ok(Self { bin: p });
            }
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(p) = exe.parent().map(|d| d.join(&name)).filter(|p| p.exists()) {
                return Ok(Self { bin: p });
            }
        }
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
                "-f",
                "bv*[ext=mp4][height<=1080]+ba[ext=m4a]/b[ext=mp4]/b",
                "--merge-output-format",
                "mp4",
                "--print",
                "after_move:filepath",
                "--no-simulate",
                "--quiet",
                "--no-warnings",
                "-o",
                &template.to_string_lossy(),
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

    /// Baixa a legenda do vídeo no idioma `lang` (ex.: `en`) para `out_dir`,
    /// preferindo a legenda enviada pelo autor e caindo para a automática do
    /// YouTube. Devolve `None` se o vídeo não tem legenda nesse idioma — isso
    /// não é erro, o whisper assume.
    pub fn download_subtitles(&self, url: &str, lang: &str, out_dir: &Path) -> Result<Option<PathBuf>> {
        std::fs::create_dir_all(out_dir)?;
        let template = out_dir.join("subs.%(ext)s");
        // Aceita variantes regionais (en-US, en-GB…) e a automática ("en-orig" etc.).
        let langs = format!("{lang},{lang}-*,{lang}.*");
        for (label, flag) in [("enviada", "--write-subs"), ("automática", "--write-auto-subs")] {
            let out = Command::new(&self.bin)
                .args([
                    "--no-playlist",
                    "--skip-download",
                    flag,
                    "--sub-langs",
                    &langs,
                    "--sub-format",
                    "vtt/srt/best",
                    "--quiet",
                    "--no-warnings",
                    "-o",
                    &template.to_string_lossy(),
                    url,
                ])
                .output()?;
            if !out.status.success() {
                let err = String::from_utf8_lossy(&out.stderr);
                tracing::debug!(%err, "yt-dlp sem legenda {label}");
                continue;
            }
            if let Some(p) = newest_subtitle(out_dir)? {
                info!(tipo = label, arquivo = %p.display(), "legenda do YouTube baixada");
                return Ok(Some(p));
            }
        }
        Ok(None)
    }

    /// Título do vídeo (para nomear a saída).
    pub fn title(&self, url: &str) -> Result<String> {
        let out =
            Command::new(&self.bin).args(["--no-playlist", "--print", "title", "--quiet", "--no-warnings", url]).output()?;
        if !out.status.success() {
            return Err(DubError::Media(String::from_utf8_lossy(&out.stderr).trim().to_string()));
        }
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    }
}

/// Arquivo de legenda (`.vtt`/`.srt`) mais recente em `dir`. O yt-dlp grava
/// como `subs.<idioma>.vtt`; se baixou mais de uma variante, a última serve.
fn newest_subtitle(dir: &Path) -> Result<Option<PathBuf>> {
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(dir)? {
        let p = entry?.path();
        let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
        if ext != "vtt" && ext != "srt" {
            continue;
        }
        let modified = p.metadata().and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
        if best.as_ref().map(|(t, _)| modified >= *t).unwrap_or(true) {
            best = Some((modified, p));
        }
    }
    Ok(best.map(|(_, p)| p))
}
