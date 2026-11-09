//! # dub-core
//!
//! Núcleo do pipeline de dublagem. Não depende de nenhuma engine concreta:
//! define os **tipos** (`Segment`, `AudioBuffer`), os **traits** que toda
//! engine implementa (`Transcriber`, `Translator`, `Synthesizer`, `Separator`,
//! `Diarizer`), a abstração de **hardware** (`Backend`) e o **pipeline**
//! paralelo por segmentos que liga tudo.
//!
//! Engines concretas vivem em outros crates (`dub-engines`, `dub-engines-py`)
//! e são injetadas via `EngineSet`.

pub mod audio;
pub mod backend;
pub mod engines;
pub mod error;
pub mod paths;
pub mod pipeline;
pub mod segment;
pub mod subtitles;
pub mod timing;

pub use audio::AudioBuffer;
pub use backend::{Backend, Device, DeviceMode, DevicePlan, GpuFit, GpuKind, Performance, Role};
pub use engines::{Diarizer, EngineSet, Separator, Synthesizer, Transcriber, Translator};
pub use error::{DubError, Result};
pub use pipeline::{ContentMode, DubJob, Pipeline, PipelineConfig, Progress, Stage};
pub use segment::{Gender, Lang, Segment, SegmentKind, Speaker};
pub use subtitles::SubtitleTranscriber;
