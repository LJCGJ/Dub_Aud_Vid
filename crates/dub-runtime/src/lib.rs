//! # dub-runtime
//!
//! O que a CLI e o app de desktop têm em comum, acima do `dub-core`:
//!
//! - [`engines`] — monta o `EngineSet` tentando engine nativa e caindo para Python;
//! - [`job`] — executa uma dublagem de vídeo do início ao fim emitindo [`JobEvent`]s;
//! - [`doctor`] — diagnóstico de hardware, ferramentas, engines e modelos.

pub mod doctor;
pub mod engines;
pub mod job;
pub mod voices;

pub use doctor::{doctor, DoctorReport, ToolStatus};
pub use engines::{build, default_stretcher, EngineChoice, Impl};
pub use job::{run_video_job, JobEvent, JobRequest, JobResult};
