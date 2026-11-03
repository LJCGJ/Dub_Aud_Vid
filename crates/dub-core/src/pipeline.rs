//! Orquestração do pipeline de dublagem.
//!
//! ```text
//!  áudio ─► [separar] ─► voz ─► [transcrever] ─► [diarizar] ─► segmentos
//!                        │                                        │
//!                      fundo                        ┌─────────────┴─────────────┐
//!                        │                          ▼                           ▼
//!                        │                 worker TRADUÇÃO ──canal──► worker SÍNTESE
//!                        │                   (CPU)                      (GPU ou CPU)
//!                        │                                                  │
//!                        └──────────────► [time-fit + mixagem] ◄────────────┘
//!                                                  │
//!                                              dub.wav
//! ```
//!
//! As etapas de tradução e síntese são encadeadas por canais: assim que um lote
//! é traduzido ele já vai para a síntese, e a tradução do lote seguinte começa.
//! Com GPU, a síntese (etapa mais pesada) fica na GPU enquanto a tradução usa a
//! CPU — é o "processamento em conjunto".

use crate::audio::AudioBuffer;
use crate::engines::EngineSet;
use crate::error::{DubError, Result};
use crate::segment::{Lang, Segment};
use crate::timing::{plan_all, TimingConfig};
use crossbeam_channel::{bounded, Sender};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use tracing::{debug, info, warn};

/// Etapas do pipeline, para progresso/UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Separate,
    Transcribe,
    Diarize,
    Translate,
    Synthesize,
    Fit,
    Mix,
    Done,
}

/// Evento de progresso emitido pelo pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Progress {
    pub stage: Stage,
    /// Itens concluídos nesta etapa (segmentos, ou 0/1 para etapas únicas).
    pub done: usize,
    pub total: usize,
    pub message: String,
}

/// Como esticar/comprimir áudio para o time-fit. `dub-media` implementa com
/// ffmpeg `atempo` (preserva o tom); o core traz um fallback simples.
pub trait Stretcher: Send + Sync {
    /// `speed > 1` encurta o áudio.
    fn stretch(&self, audio: &AudioBuffer, speed: f32) -> Result<AudioBuffer>;
}

/// Fallback: reamostragem linear. **Muda o tom** — só para quando não há ffmpeg.
pub struct NaiveStretcher;

impl Stretcher for NaiveStretcher {
    fn stretch(&self, audio: &AudioBuffer, speed: f32) -> Result<AudioBuffer> {
        if (speed - 1.0).abs() < 1e-3 || audio.is_empty() {
            return Ok(audio.clone());
        }
        let n_out = (audio.samples.len() as f32 / speed).round().max(1.0) as usize;
        let mut out = Vec::with_capacity(n_out);
        for i in 0..n_out {
            let pos = i as f32 * speed;
            let i0 = pos.floor() as usize;
            let frac = pos - i0 as f32;
            let a = audio.samples.get(i0).copied().unwrap_or(0.0);
            let b = audio.samples.get(i0 + 1).copied().unwrap_or(a);
            out.push(a + (b - a) * frac);
        }
        Ok(AudioBuffer::new(audio.sample_rate, out))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub source_lang: Option<Lang>,
    pub target_lang: Lang,
    /// Tamanho do lote enviado ao tradutor (e ao canal de síntese).
    pub translate_batch: usize,
    /// Usar separação voz/fundo se houver `Separator`.
    pub separate: bool,
    /// Usar diarização se houver `Diarizer`.
    pub diarize: bool,
    /// Volume do fundo original na mixagem final (0..1).
    pub background_gain: f32,
    /// Volume da voz dublada (0..1).
    pub voice_gain: f32,
    /// Em trechos sobrepostos, reduz a voz dublada por este fator (fica mais
    /// natural do que duas dublagens brigando).
    pub overlap_gain: f32,
    pub timing: TimingConfig,
    /// Sample rate da trilha final.
    pub sample_rate: u32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            source_lang: Some(Lang::en()),
            target_lang: Lang::pt_br(),
            translate_batch: 8,
            separate: true,
            diarize: true,
            background_gain: 0.9,
            voice_gain: 1.0,
            overlap_gain: 0.6,
            timing: TimingConfig::default(),
            sample_rate: 44_100,
        }
    }
}

/// Entrada de um trabalho de dublagem.
pub struct DubJob {
    /// WAV mono 16 kHz do áudio original completo.
    pub audio: PathBuf,
    /// Diretório de trabalho (arquivos intermediários).
    pub work_dir: PathBuf,
}

/// Saída do pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DubOutput {
    /// Trilha final (voz dublada + fundo), pronta para remux.
    pub mixed_wav: PathBuf,
    /// Só a voz dublada, alinhada no tempo.
    pub voice_wav: PathBuf,
    /// Fundo separado (None se não houve separação).
    pub background_wav: Option<PathBuf>,
    pub segments: Vec<Segment>,
    /// Legendas no idioma de destino (SRT), geradas de brinde.
    pub srt: PathBuf,
}

pub struct Pipeline {
    engines: EngineSet,
    stretcher: Arc<dyn Stretcher>,
    config: PipelineConfig,
    cancel: Arc<AtomicBool>,
}

impl Pipeline {
    pub fn new(engines: EngineSet, config: PipelineConfig) -> Self {
        Self { engines, stretcher: Arc::new(NaiveStretcher), config, cancel: Arc::new(AtomicBool::new(false)) }
    }

    pub fn with_stretcher(mut self, s: Arc<dyn Stretcher>) -> Self {
        self.stretcher = s;
        self
    }

    /// Handle para cancelar de outra thread.
    pub fn cancel_handle(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }

    fn check_cancel(&self) -> Result<()> {
        if self.cancel.load(Ordering::Relaxed) {
            Err(DubError::Cancelled)
        } else {
            Ok(())
        }
    }

    /// Executa o pipeline completo. `progress` recebe eventos; pode ser um canal
    /// com buffer grande ou `None`.
    pub fn run(&self, job: &DubJob, progress: Option<Sender<Progress>>) -> Result<DubOutput> {
        std::fs::create_dir_all(&job.work_dir)?;
        let report = |stage: Stage, done: usize, total: usize, msg: &str| {
            if let Some(p) = &progress {
                let _ = p.send(Progress { stage, done, total, message: msg.to_string() });
            }
        };

        for (role, info) in self.engines.describe() {
            info!(role, engine = %info.name, model = %info.model, device = %info.device, "engine");
        }

        // 1. Separação ---------------------------------------------------------
        let (voice_path, background_path) = match (&self.engines.separator, self.config.separate) {
            (Some(sep), true) => {
                report(Stage::Separate, 0, 1, "separando voz e fundo");
                let (v, b) = sep.separate(&job.audio, &job.work_dir)?;
                report(Stage::Separate, 1, 1, "separação concluída");
                (v, Some(b))
            }
            _ => (job.audio.clone(), None),
        };
        self.check_cancel()?;

        // 2. Transcrição -------------------------------------------------------
        report(Stage::Transcribe, 0, 1, "transcrevendo");
        let mut segments = self.engines.transcriber.transcribe(&voice_path, self.config.source_lang.as_ref())?;
        segments.retain(|s| !s.source_text.trim().is_empty() && s.end > s.start);
        for (i, s) in segments.iter_mut().enumerate() {
            s.index = i;
        }
        report(Stage::Transcribe, 1, 1, &format!("{} segmentos", segments.len()));
        self.check_cancel()?;

        // 3. Diarização --------------------------------------------------------
        if let (Some(dia), true) = (&self.engines.diarizer, self.config.diarize) {
            report(Stage::Diarize, 0, 1, "identificando falantes");
            segments = dia.diarize(&voice_path, segments)?;
            let n_spk = dia.speakers().len();
            report(Stage::Diarize, 1, 1, &format!("{n_spk} falantes"));
        }
        self.check_cancel()?;

        if segments.is_empty() {
            return Err(DubError::Audio("nenhuma fala detectada".into()));
        }

        // 4. Tradução ║ Síntese (paralelo) -------------------------------------
        let total = segments.len();
        let synth_dir = job.work_dir.join("tts");
        std::fs::create_dir_all(&synth_dir)?;

        let voice_audio = if self.engines.synthesizer.supports_voice_cloning() {
            Some(Arc::new(AudioBuffer::read_wav(&voice_path)?))
        } else {
            None
        };

        let segments = self.translate_and_synthesize(segments, &synth_dir, voice_audio, &progress)?;
        self.check_cancel()?;
        report(Stage::Synthesize, total, total, "síntese concluída");

        // 5. Time-fit + 6. Mixagem ---------------------------------------------
        let output = self.fit_and_mix(segments, background_path, &job.work_dir, &report)?;
        report(Stage::Done, 1, 1, "concluído");
        Ok(output)
    }

    fn translate_and_synthesize(
        &self,
        segments: Vec<Segment>,
        synth_dir: &Path,
        voice_audio: Option<Arc<AudioBuffer>>,
        progress: &Option<Sender<Progress>>,
    ) -> Result<Vec<Segment>> {
        let total = segments.len();
        let batch = self.config.translate_batch.max(1);
        // Buffer de 2 lotes: tradução pode correr um pouco à frente da síntese.
        let (tx_translated, rx_translated) = bounded::<Vec<Segment>>(2);
        let (tx_done, rx_done) = bounded::<Result<Segment>>(total.max(1));

        let translator = self.engines.translator.clone();
        let synthesizer = self.engines.synthesizer.clone();
        let cfg = self.config.clone();
        let cancel = self.cancel.clone();
        let progress_t = progress.clone();
        let progress_s = progress.clone();
        let synth_dir = synth_dir.to_path_buf();

        // Worker de tradução (CPU) ------------------------------------------
        let t_handle = thread::Builder::new().name("dub-translate".into()).spawn(move || -> Result<()> {
            let mut done = 0usize;
            for chunk in segments.chunks(batch) {
                if cancel.load(Ordering::Relaxed) {
                    return Err(DubError::Cancelled);
                }
                let texts: Vec<&str> = chunk.iter().map(|s| s.source_text.as_str()).collect();
                let from = cfg.source_lang.clone().unwrap_or_else(Lang::en);
                let translated = translator.translate(&texts, &from, &cfg.target_lang)?;
                if translated.len() != chunk.len() {
                    return Err(DubError::EngineFailed {
                        engine: translator.info().name,
                        stage: "translate",
                        message: format!("esperava {} traduções, recebeu {}", chunk.len(), translated.len()),
                    });
                }
                let mut out = chunk.to_vec();
                for (s, t) in out.iter_mut().zip(translated) {
                    s.target_text = Some(t);
                }
                done += out.len();
                if let Some(p) = &progress_t {
                    let _ = p.send(Progress { stage: Stage::Translate, done, total, message: String::new() });
                }
                tx_translated.send(out).map_err(|_| DubError::ChannelClosed("translated"))?;
            }
            Ok(())
        })?;

        // Worker de síntese (GPU se houver) --------------------------------
        let cancel_s = self.cancel.clone();
        let cfg_s = self.config.clone();
        let s_handle = thread::Builder::new().name("dub-synth".into()).spawn(move || -> Result<()> {
            let mut done = 0usize;
            let use_ref = synthesizer.supports_voice_cloning();
            let pass_speed = synthesizer.supports_speed();
            for chunk in rx_translated.iter() {
                for mut seg in chunk {
                    if cancel_s.load(Ordering::Relaxed) {
                        return Err(DubError::Cancelled);
                    }
                    let text = seg.target_text.clone().unwrap_or_default();
                    let result = (|| -> Result<Segment> {
                        // Referência de voz: o próprio trecho original do falante.
                        let ref_path = if use_ref {
                            voice_audio.as_ref().and_then(|va| {
                                let p = synth_dir.join(format!("ref_{:05}.wav", seg.index));
                                slice(va, seg.start, seg.end).write_wav(&p).ok().map(|_| p)
                            })
                        } else {
                            None
                        };
                        // Primeira estimativa de velocidade: proporção de caracteres
                        // (barato e já evita um segundo passe de TTS na maioria dos casos).
                        let speed_hint = if pass_speed {
                            estimate_speed(&seg.source_text, &text, &cfg_s.timing)
                        } else {
                            1.0
                        };
                        let audio = synthesizer.synthesize(&text, &cfg_s.target_lang, ref_path.as_deref(), speed_hint)?;
                        let p = synth_dir.join(format!("seg_{:05}.wav", seg.index));
                        audio.write_wav(&p)?;
                        seg.dubbed_wav = Some(p);
                        Ok(seg)
                    })();
                    done += 1;
                    if let Some(p) = &progress_s {
                        let _ = p.send(Progress { stage: Stage::Synthesize, done, total, message: String::new() });
                    }
                    tx_done.send(result).map_err(|_| DubError::ChannelClosed("done"))?;
                }
            }
            Ok(())
        })?;

        // Coleta ------------------------------------------------------------
        let mut out: Vec<Segment> = Vec::with_capacity(total);
        let mut first_err: Option<DubError> = None;
        for r in rx_done.iter() {
            match r {
                Ok(s) => out.push(s),
                Err(e) => {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                    self.cancel.store(true, Ordering::Relaxed);
                }
            }
            if out.len() + first_err.iter().count() >= total {
                break;
            }
        }
        // Encerra workers (ignora erros de canal fechado após falha).
        let t_res = t_handle.join().map_err(|_| DubError::ChannelClosed("translate thread"))?;
        let s_res = s_handle.join().map_err(|_| DubError::ChannelClosed("synth thread"))?;
        if let Some(e) = first_err {
            return Err(e);
        }
        t_res?;
        s_res?;

        out.sort_by_key(|s| s.index);
        Ok(out)
    }

    fn fit_and_mix(
        &self,
        mut segments: Vec<Segment>,
        background: Option<PathBuf>,
        work_dir: &Path,
        report: &dyn Fn(Stage, usize, usize, &str),
    ) -> Result<DubOutput> {
        let sr = self.config.sample_rate;
        let total = segments.len();

        // Carrega as dublagens e mede.
        report(Stage::Fit, 0, total, "ajustando tempo");
        let mut dubbed: Vec<AudioBuffer> = Vec::with_capacity(total);
        for s in &segments {
            let p = s.dubbed_wav.as_ref().ok_or_else(|| DubError::Audio(format!("segmento {} sem áudio", s.index)))?;
            let a = AudioBuffer::read_wav(p)?;
            dubbed.push(resample_if_needed(a, sr));
        }
        let lens: Vec<f64> = dubbed.iter().map(|a| a.duration()).collect();
        let plans = plan_all(&mut segments, &lens, &self.config.timing);

        // Estica e posiciona.
        let total_len = background
            .as_ref()
            .and_then(|b| AudioBuffer::read_wav(b).ok())
            .map(|b| b.duration())
            .unwrap_or_else(|| segments.last().map(|s| s.end + 2.0).unwrap_or(0.0));
        let mut voice_track = AudioBuffer::silence(sr, total_len);

        for (i, (seg, plan)) in segments.iter().zip(&plans).enumerate() {
            self.check_cancel()?;
            let a = if (plan.speed - 1.0).abs() > 0.01 {
                self.stretcher.stretch(&dubbed[i], plan.speed)?
            } else {
                dubbed[i].clone()
            };
            if plan.overflow > 0.3 {
                warn!(index = seg.index, overflow = plan.overflow, "dublagem transborda além da janela");
            }
            let gain = if seg.overlapped { self.config.voice_gain * self.config.overlap_gain } else { self.config.voice_gain };
            voice_track.mix_at(&a, seg.start, gain)?;
            debug!(index = seg.index, speed = plan.speed, "segmento posicionado");
            report(Stage::Fit, i + 1, total, "");
        }

        let voice_wav = work_dir.join("voice_dubbed.wav");
        voice_track.limit_peak(0.95);
        voice_track.write_wav(&voice_wav)?;

        // Mixagem ----------------------------------------------------------
        report(Stage::Mix, 0, 1, "mixando");
        let mut mixed = voice_track.clone();
        if let Some(bg) = &background {
            let bg_audio = resample_if_needed(AudioBuffer::read_wav(bg)?, sr);
            mixed.mix_at(&bg_audio, 0.0, self.config.background_gain)?;
        }
        mixed.limit_peak(0.95);
        let mixed_wav = work_dir.join("dub_mix.wav");
        mixed.write_wav(&mixed_wav)?;

        let srt = work_dir.join("dub.srt");
        std::fs::write(&srt, to_srt(&segments))?;
        report(Stage::Mix, 1, 1, "");

        Ok(DubOutput { mixed_wav, voice_wav, background_wav: background, segments, srt })
    }
}

// --- utilitários --------------------------------------------------------------

fn slice(a: &AudioBuffer, start: f64, end: f64) -> AudioBuffer {
    let sr = a.sample_rate as f64;
    let i0 = ((start * sr) as usize).min(a.samples.len());
    let i1 = ((end * sr) as usize).min(a.samples.len());
    AudioBuffer::new(a.sample_rate, a.samples[i0..i1].to_vec())
}

/// Estimativa grosseira: português tem ~20% mais caracteres que inglês para o
/// mesmo conteúdo; se o texto-alvo cresceu mais que isso, pede fala mais rápida.
fn estimate_speed(source: &str, target: &str, cfg: &TimingConfig) -> f32 {
    let s = source.chars().count().max(1) as f32;
    let t = target.chars().count().max(1) as f32;
    let ratio = t / (s * 1.2);
    ratio.clamp(1.0, cfg.max_speed)
}

/// Reamostragem linear simples para alinhar sample rate antes da mixagem.
fn resample_if_needed(a: AudioBuffer, target_sr: u32) -> AudioBuffer {
    if a.sample_rate == target_sr || a.is_empty() {
        return AudioBuffer { sample_rate: target_sr, ..a };
    }
    let ratio = a.sample_rate as f64 / target_sr as f64;
    let n_out = (a.samples.len() as f64 / ratio).round() as usize;
    let mut out = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let pos = i as f64 * ratio;
        let i0 = pos.floor() as usize;
        let frac = (pos - i0 as f64) as f32;
        let x0 = a.samples.get(i0).copied().unwrap_or(0.0);
        let x1 = a.samples.get(i0 + 1).copied().unwrap_or(x0);
        out.push(x0 + (x1 - x0) * frac);
    }
    AudioBuffer::new(target_sr, out)
}

fn srt_time(t: f64) -> String {
    let ms = (t * 1000.0).round() as u64;
    format!("{:02}:{:02}:{:02},{:03}", ms / 3_600_000, (ms / 60_000) % 60, (ms / 1000) % 60, ms % 1000)
}

pub fn to_srt(segments: &[Segment]) -> String {
    let mut out = String::new();
    for (i, s) in segments.iter().enumerate() {
        let text = s.target_text.as_deref().unwrap_or(&s.source_text);
        out.push_str(&format!("{}\n{} --> {}\n{}\n\n", i + 1, srt_time(s.start), srt_time(s.end), text));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::*;

    struct Fake;
    impl Engine for Fake {
        fn info(&self) -> EngineInfo {
            EngineInfo { name: "fake".into(), model: "-".into(), device: "cpu".into() }
        }
    }
    impl Transcriber for Fake {
        fn transcribe(&self, _: &Path, _: Option<&Lang>) -> Result<Vec<Segment>> {
            Ok(vec![Segment::new(0, 0.5, 1.5, "hello world"), Segment::new(1, 2.0, 3.0, "second line")])
        }
    }
    impl Translator for Fake {
        fn translate(&self, texts: &[&str], _: &Lang, _: &Lang) -> Result<Vec<String>> {
            Ok(texts.iter().map(|t| format!("PT[{t}]")).collect())
        }
    }
    impl Synthesizer for Fake {
        fn synthesize(&self, text: &str, _: &Lang, _: Option<&Path>, _: f32) -> Result<AudioBuffer> {
            // 0.1s por palavra, tom constante
            let secs = 0.1 * text.split_whitespace().count() as f64 + 0.5;
            let n = (16000.0 * secs) as usize;
            Ok(AudioBuffer::new(16000, (0..n).map(|i| ((i as f32) * 0.05).sin() * 0.3).collect()))
        }
    }

    #[test]
    fn end_to_end_with_fakes() {
        let dir = std::env::temp_dir().join(format!("dub-pipe-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("in.wav");
        AudioBuffer::silence(16000, 4.0).write_wav(&src).unwrap();

        let f = Arc::new(Fake);
        let engines = EngineSet {
            transcriber: f.clone(),
            translator: f.clone(),
            synthesizer: f.clone(),
            separator: None,
            diarizer: None,
        };
        let cfg = PipelineConfig { sample_rate: 16000, ..Default::default() };
        let (tx, rx) = crossbeam_channel::unbounded();
        let out = Pipeline::new(engines, cfg).run(&DubJob { audio: src, work_dir: dir.clone() }, Some(tx)).unwrap();

        assert_eq!(out.segments.len(), 2);
        assert_eq!(out.segments[0].target_text.as_deref(), Some("PT[hello world]"));
        assert!(out.mixed_wav.exists() && out.srt.exists());
        let mixed = AudioBuffer::read_wav(&out.mixed_wav).unwrap();
        assert!(mixed.duration() >= 3.0);
        let events: Vec<_> = rx.try_iter().collect();
        assert!(events.iter().any(|e| e.stage == Stage::Done));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn srt_format() {
        let s = to_srt(&[Segment::new(0, 1.5, 2.25, "x")]);
        assert!(s.starts_with("1\n00:00:01,500 --> 00:00:02,250\nx\n"));
    }
}
