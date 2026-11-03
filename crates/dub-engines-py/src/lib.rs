//! # dub-engines-py
//!
//! Implementa os traits de `dub-core` por cima de processos Python
//! (`python -m dub_engines <engine>`). Cada engine é um processo próprio, então
//! o whisper pode estar na GPU enquanto o tradutor roda na CPU — e se um
//! processo morrer, só aquela etapa falha.
//!
//! O core em Rust **nunca** depende disto para compilar ou rodar: é um plugin.

pub mod process;
pub mod python;
pub mod wrappers;

pub use process::{PyProcess, PySpec};
pub use python::PythonEnv;
pub use wrappers::{PyDiarizer, PySeparator, PySynthesizer, PyTranscriber, PyTranslator};
