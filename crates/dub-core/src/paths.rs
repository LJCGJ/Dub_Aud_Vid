//! Onde modelos e dados ficam no disco. Compartilhado por engines nativas e
//! pelo gerenciador de modelos, para que `dub models download` grave exatamente
//! onde as engines procuram.
//!
//! Ordem de busca (a primeira existente ganha para leitura; para escrita, a
//! primeira gravável):
//! 1. `DUB_MODELS_DIR`
//! 2. `<pasta do executável>/models` — layout portátil (zip do release)
//! 3. `~/.cache/dub` (Linux/macOS) ou `%LOCALAPPDATA%\dub` (Windows)

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    Whisper,
    Voice,
    Translate,
    /// Separação voz/instrumental (MDX-Net).
    Separate,
    /// Modelo + vozes do Kokoro.
    Kokoro,
}

impl ModelKind {
    pub fn subdir(self) -> &'static str {
        match self {
            ModelKind::Whisper => "whisper",
            ModelKind::Voice => "voices",
            ModelKind::Translate => "translate",
            ModelKind::Separate => "separate",
            ModelKind::Kokoro => "kokoro",
        }
    }
    pub fn all() -> [ModelKind; 5] {
        [ModelKind::Whisper, ModelKind::Voice, ModelKind::Translate, ModelKind::Separate, ModelKind::Kokoro]
    }
}

impl std::fmt::Display for ModelKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.subdir())
    }
}

/// Raízes candidatas, em ordem de prioridade.
pub fn model_roots() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(d) = std::env::var_os("DUB_MODELS_DIR") {
        v.push(PathBuf::from(d));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            v.push(dir.join("models"));
        }
    }
    if let Some(cache) = user_cache_dir() {
        v.push(cache);
    }
    v
}

/// Pastas onde procurar modelos de um tipo (todas as raízes + subpasta).
pub fn model_dirs(kind: ModelKind) -> Vec<PathBuf> {
    model_roots().into_iter().map(|r| r.join(kind.subdir())).collect()
}

/// Pasta onde **gravar** novos modelos: a primeira raiz que já existe e é
/// gravável; senão, o cache do usuário.
pub fn install_dir(kind: ModelKind) -> PathBuf {
    for root in model_roots() {
        if root.is_dir() && is_writable(&root) {
            return root.join(kind.subdir());
        }
    }
    user_cache_dir().unwrap_or_else(|| PathBuf::from("models")).join(kind.subdir())
}

fn user_cache_dir() -> Option<PathBuf> {
    if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").map(|p| PathBuf::from(p).join("dub"))
    } else {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
            .map(|c| c.join("dub"))
    }
}

fn is_writable(dir: &std::path::Path) -> bool {
    let probe = dir.join(".dub-write-test");
    match std::fs::write(&probe, b"") {
        Ok(()) => {
            let _ = std::fs::remove_file(probe);
            true
        }
        Err(_) => false,
    }
}

/// Procura um arquivo (ou pasta) pelo nome em todas as pastas do tipo.
pub fn find(kind: ModelKind, name: &str) -> Option<PathBuf> {
    model_dirs(kind).into_iter().map(|d| d.join(name)).find(|p| p.exists())
}
