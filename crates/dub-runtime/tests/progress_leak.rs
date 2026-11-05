//! Regressão: uma engine que guarda o callback de progresso para sempre (como o
//! whisper-rs faz) não pode travar o fim do job. A versão 0.1.2 travava em
//! "mixando" com tudo pronto, porque o canal de progresso nunca fechava.

use dub_core::engines::*;
use dub_core::{AudioBuffer, Lang, Pipeline, PipelineConfig, Progress, Segment};
use std::path::Path;
use std::sync::Arc;

struct Leaky;
impl Engine for Leaky {
    fn info(&self) -> EngineInfo {
        EngineInfo { name: "leaky".into(), model: "-".into(), device: "cpu".into() }
    }
}
impl Transcriber for Leaky {
    fn transcribe(&self, _: &Path, _: Option<&Lang>) -> dub_core::Result<Vec<Segment>> {
        Ok(vec![Segment::new(0, 0.0, 1.0, "hello")])
    }
    fn transcribe_with_progress(
        &self,
        audio: &Path,
        lang: Option<&Lang>,
        on_progress: Arc<dyn Fn(u8) + Send + Sync>,
    ) -> dub_core::Result<Vec<Segment>> {
        on_progress(50);
        std::mem::forget(on_progress); // vaza, como o whisper-rs
        self.transcribe(audio, lang)
    }
}
impl Translator for Leaky {
    fn translate(&self, texts: &[&str], _: &Lang, _: &Lang) -> dub_core::Result<Vec<String>> {
        Ok(texts.iter().map(|t| format!("olá {t}")).collect())
    }
}
impl Synthesizer for Leaky {
    fn synthesize(&self, _: &str, _: &Lang, _: Option<&Path>, _: f32) -> dub_core::Result<AudioBuffer> {
        Ok(AudioBuffer::new(16_000, vec![0.1; 8000]))
    }
}

#[test]
fn job_finishes_even_if_engine_keeps_progress_callback() {
    let dir = tempfile::tempdir().unwrap();
    let audio = dir.path().join("in.wav");
    AudioBuffer::new(16_000, vec![0.05; 32_000]).write_wav(&audio).unwrap();
    let job = dub_core::DubJob { audio, work_dir: dir.path().join("work") };
    let engines = EngineSet {
        transcriber: Arc::new(Leaky),
        translator: Arc::new(Leaky),
        synthesizer: Arc::new(Leaky),
        separator: None,
        diarizer: None,
    };
    let pipeline = Pipeline::new(engines, PipelineConfig::default());

    // Mesmo laço de eventos do run_video_job.
    let (tx, rx) = crossbeam_channel::unbounded::<Progress>();
    let started = std::time::Instant::now();
    let out = std::thread::scope(|s| {
        let h = s.spawn(|| pipeline.run(&job, Some(tx)));
        let mut n = 0;
        loop {
            match rx.recv_timeout(std::time::Duration::from_millis(100)) {
                Ok(_) => n += 1,
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    if h.is_finished() {
                        break;
                    }
                    assert!(started.elapsed().as_secs() < 30, "job travou esperando o canal de progresso");
                }
            }
        }
        assert!(n > 0);
        h.join().unwrap()
    });
    assert!(out.is_ok(), "{:?}", out.err());
}
