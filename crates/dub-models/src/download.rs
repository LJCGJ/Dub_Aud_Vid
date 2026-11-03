use crate::catalog::ModelEntry;
use anyhow::{anyhow, bail, Context, Result};
use dub_core::paths::{self, ModelKind};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tracing::info;

/// Progresso de um download, para a CLI/UI desenharem a barra.
#[derive(Debug, Clone)]
pub struct Progress {
    pub file: String,
    pub downloaded: u64,
    pub total: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstallStatus {
    Installed(PathBuf),
    Missing,
}

/// Caminho onde a entrada ficaria/está instalada (arquivo principal ou pasta).
pub fn installed_path(entry: &ModelEntry) -> Option<PathBuf> {
    let main = entry.files.first()?;
    let target_name = if main.archive { entry.name.clone() } else { main.name.clone() };
    paths::find(entry.kind, &target_name)
}

pub fn status(entry: &ModelEntry) -> InstallStatus {
    match installed_path(entry) {
        Some(p) => InstallStatus::Installed(p),
        None => InstallStatus::Missing,
    }
}

/// Baixa e instala todos os arquivos da entrada. Idempotente: pula o que já existe.
pub fn install(entry: &ModelEntry, mut on_progress: impl FnMut(Progress)) -> Result<PathBuf> {
    let dir = paths::install_dir(entry.kind);
    std::fs::create_dir_all(&dir).with_context(|| format!("criar {}", dir.display()))?;

    let mut result = dir.clone();
    for f in &entry.files {
        if f.archive {
            let dest_dir = dir.join(&entry.name);
            if dest_dir.join("model.bin").exists() || dest_dir.join("config.json").exists() {
                info!(path = %dest_dir.display(), "já instalado");
                result = dest_dir;
                continue;
            }
            let tmp = dir.join(format!("{}.part", f.name));
            fetch(&f.url, &tmp, &f.name, &mut on_progress)?;
            extract_tar_gz(&tmp, &dest_dir).with_context(|| format!("extrair {}", f.name))?;
            let _ = std::fs::remove_file(&tmp);
            result = dest_dir;
        } else {
            let dest = dir.join(&f.name);
            if dest.exists() && std::fs::metadata(&dest).map(|m| m.len() > 0).unwrap_or(false) {
                info!(path = %dest.display(), "já instalado");
            } else {
                let tmp = dir.join(format!("{}.part", f.name));
                fetch(&f.url, &tmp, &f.name, &mut on_progress)?;
                std::fs::rename(&tmp, &dest)?;
            }
            if entry.files.first().map(|m| m.name == f.name).unwrap_or(false) {
                result = dest;
            }
        }
    }
    Ok(result)
}

fn fetch(url: &str, dest: &Path, label: &str, on_progress: &mut impl FnMut(Progress)) -> Result<()> {
    info!(url, "baixando");
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(30))
        .timeout_read(Duration::from_secs(120))
        .redirects(8)
        .user_agent(concat!("dub/", env!("CARGO_PKG_VERSION")))
        .build();
    let resp = agent.get(url).call().map_err(|e| match e {
        ureq::Error::Status(404, _) => anyhow!("{label}: não encontrado (404) em {url}"),
        ureq::Error::Status(code, _) => anyhow!("{label}: HTTP {code} em {url}"),
        other => anyhow!("{label}: {other}"),
    })?;
    let total = resp.header("Content-Length").and_then(|v| v.parse::<u64>().ok());
    let mut reader = resp.into_reader();
    let mut out = File::create(dest).with_context(|| format!("criar {}", dest.display()))?;
    let mut buf = vec![0u8; 1 << 16];
    let mut downloaded = 0u64;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        downloaded += n as u64;
        on_progress(Progress { file: label.to_string(), downloaded, total });
    }
    out.flush()?;
    if let Some(t) = total {
        if downloaded != t {
            let _ = std::fs::remove_file(dest);
            bail!("{label}: download incompleto ({downloaded} de {t} bytes)");
        }
    }
    if downloaded < 1024 {
        // Páginas de erro do Hugging Face vêm como HTML pequeno com status 200.
        let _ = std::fs::remove_file(dest);
        bail!("{label}: resposta suspeita ({downloaded} bytes) — URL inválida ou modelo removido");
    }
    Ok(())
}

fn extract_tar_gz(archive: &Path, dest_dir: &Path) -> Result<()> {
    let file = File::open(archive)?;
    let gz = flate2::read::GzDecoder::new(file);
    let mut tar = tar::Archive::new(gz);
    std::fs::create_dir_all(dest_dir)?;
    // Achata um possível diretório raiz único dentro do tar.
    let tmp = dest_dir.with_extension("extracting");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp)?;
    tar.unpack(&tmp)?;
    let entries: Vec<_> = std::fs::read_dir(&tmp)?.flatten().collect();
    let src_root = if entries.len() == 1 && entries[0].file_type().map(|t| t.is_dir()).unwrap_or(false) {
        entries[0].path()
    } else {
        tmp.clone()
    };
    for e in std::fs::read_dir(&src_root)?.flatten() {
        let to = dest_dir.join(e.file_name());
        let _ = std::fs::remove_dir_all(&to);
        let _ = std::fs::remove_file(&to);
        std::fs::rename(e.path(), &to)?;
    }
    let _ = std::fs::remove_dir_all(&tmp);
    Ok(())
}

/// Resumo por tipo para `dub models list` e `dub doctor`.
pub fn summary() -> Vec<(ModelKind, Vec<(ModelEntry, InstallStatus)>)> {
    ModelKind::all()
        .into_iter()
        .map(|k| (k, crate::catalog().into_iter().filter(|m| m.kind == k).map(|m| (m.clone(), status(&m))).collect()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_names_are_unique_and_urls_well_formed() {
        let cat = crate::catalog();
        let mut names: Vec<_> = cat.iter().map(|m| m.name.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), cat.len());
        for m in &cat {
            for f in &m.files {
                assert!(f.url.starts_with("https://"), "{}", f.url);
                assert!(f.url.ends_with(&f.name), "{} não termina com {}", f.url, f.name);
            }
        }
    }

    #[test]
    fn defaults_cover_every_kind() {
        let d = crate::defaults();
        for k in ModelKind::all() {
            assert!(d.iter().any(|m| m.kind == k), "sem padrão para {k}");
        }
    }
}
