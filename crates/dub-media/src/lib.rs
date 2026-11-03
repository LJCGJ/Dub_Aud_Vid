//! # dub-media
//!
//! Tudo que toca arquivos de mídia reais: extrair áudio de vídeo, baixar do
//! YouTube, esticar áudio preservando o tom e devolver a dublagem para dentro
//! do vídeo. Usa `ffmpeg`/`ffprobe`/`yt-dlp` como processos externos — sem
//! linkar libav, o que mantém o build simples em qualquer plataforma.

pub mod ffmpeg;
pub mod stretch;
pub mod youtube;

pub use ffmpeg::{Ffmpeg, MediaInfo};
pub use stretch::FfmpegStretcher;
pub use youtube::YtDlp;
