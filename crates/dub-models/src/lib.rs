//! # dub-models
//!
//! Catálogo e download de modelos para `~/.cache/dub` (ou a pasta `models/` ao
//! lado do executável, no layout portátil). As engines nativas procuram nos
//! mesmos lugares (`dub_core::paths`), então `dub models download` deixa tudo
//! pronto para `dub video` sem configurar nada.

pub mod catalog;
pub mod download;

pub use catalog::{catalog, defaults, find, ModelEntry};
pub use download::{install, installed_path, status, InstallStatus, Progress};
