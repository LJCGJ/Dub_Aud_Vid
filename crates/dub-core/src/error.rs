use thiserror::Error;

pub type Result<T> = std::result::Result<T, DubError>;

#[derive(Debug, Error)]
pub enum DubError {
    #[error("engine '{engine}' indisponível: {reason}")]
    EngineUnavailable { engine: String, reason: String },

    #[error("falha na engine '{engine}' durante {stage}: {message}")]
    EngineFailed { engine: String, stage: &'static str, message: String },

    #[error("ferramenta externa não encontrada: {0}")]
    ToolMissing(String),

    #[error("erro de áudio: {0}")]
    Audio(String),

    #[error("erro de mídia (ffmpeg): {0}")]
    Media(String),

    #[error("pipeline cancelado")]
    Cancelled,

    #[error("canal interno fechado inesperadamente ({0})")]
    ChannelClosed(&'static str),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
