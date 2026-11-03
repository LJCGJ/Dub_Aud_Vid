use crate::ffmpeg::Ffmpeg;
use dub_core::pipeline::Stretcher;
use dub_core::{AudioBuffer, Result};

/// `Stretcher` que usa o filtro `atempo` do ffmpeg — altera a duração
/// **sem mudar o tom** da voz. É o que o pipeline deve usar sempre que ffmpeg
/// estiver disponível.
pub struct FfmpegStretcher {
    ff: Ffmpeg,
    tmp: tempfile::TempDir,
}

impl FfmpegStretcher {
    pub fn new(ff: Ffmpeg) -> Result<Self> {
        Ok(Self { ff, tmp: tempfile::tempdir()? })
    }
}

impl Stretcher for FfmpegStretcher {
    fn stretch(&self, audio: &AudioBuffer, speed: f32) -> Result<AudioBuffer> {
        if (speed - 1.0).abs() < 1e-3 || audio.is_empty() {
            return Ok(audio.clone());
        }
        let id = uuid_like();
        let input = self.tmp.path().join(format!("in_{id}.wav"));
        let output = self.tmp.path().join(format!("out_{id}.wav"));
        audio.write_wav(&input)?;
        self.ff.atempo(&input, &output, speed)?;
        let out = AudioBuffer::read_wav(&output)?;
        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
        Ok(out)
    }
}

fn uuid_like() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    format!("{}_{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stretch_changes_duration() {
        let Ok(ff) = Ffmpeg::locate() else { return };
        let s = FfmpegStretcher::new(ff).unwrap();
        let a = AudioBuffer::new(16000, (0..32000).map(|i| ((i as f32) * 0.05).sin() * 0.5).collect());
        let b = s.stretch(&a, 1.25).unwrap();
        let expected = a.duration() / 1.25;
        assert!((b.duration() - expected).abs() < 0.05, "got {} want {}", b.duration(), expected);
    }
}
