//! # dub-native
//!
//! Engines nativas em Rust + C++, atrás dos mesmos traits de `dub-core`:
//!
//! - [`NativeStretcher`] — time-stretch WSOLA em C++ (`native/timestretch`), sempre disponível.
//! - [`WhisperTranscriber`] — whisper.cpp via `whisper-rs` (feature `whisper`).
//! - [`PiperSynthesizer`] — Piper (VITS) via ONNX Runtime + espeak-ng (feature `piper`).
//! - [`Ct2Translator`] — Marian/OPUS-MT via CTranslate2 (feature `translate`).
//!
//! GPU: whisper.cpp usa o backend escolhido em tempo de compilação
//! (`whisper-cuda` / `whisper-vulkan` / `whisper-metal`); o ONNX Runtime é
//! carregado dinamicamente, então basta trocar a `onnxruntime.dll` pela build
//! DirectML ou CUDA para a síntese usar a GPU — sem recompilar.

pub mod stretch;
pub use stretch::NativeStretcher;

#[cfg(feature = "whisper")]
pub mod whisper;
#[cfg(feature = "whisper")]
pub use whisper::WhisperTranscriber;

#[cfg(feature = "piper")]
pub mod phonemize;
#[cfg(feature = "piper")]
pub mod piper;
#[cfg(feature = "piper")]
pub use piper::{PiperSynthesizer, VoicePack};

#[cfg(feature = "translate")]
pub mod translate;
#[cfg(feature = "translate")]
pub use translate::Ct2Translator;

#[cfg(feature = "separate")]
pub mod separate;
#[cfg(feature = "separate")]
pub use separate::MdxSeparator;
