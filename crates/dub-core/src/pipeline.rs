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
use crate::segment::{Lang, Segment, Speaker};
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
    /// Estimar se cada voz original é masculina ou feminina (pelo tom), para
    /// engines que escolhem a voz da dublagem por trecho/falante.
    pub detect_gender: bool,
    /// Com voz isolada, ajustar início/fim de cada trecho ao instante em que a
    /// voz original realmente soa (o whisper marca com folga, sobretudo em canto).
    pub align_to_voice: bool,
    /// Aproximar a dublagem da voz original: tom (até ±4 semitons) e brilho.
    pub match_voice: bool,
    /// Volume do fundo original na mixagem final (0..1).
    pub background_gain: f32,
    /// Sem separador de voz/fundo, o áudio original inteiro vira o fundo e é
    /// abaixado para este volume só enquanto a voz dublada toca (*ducking*);
    /// música e efeitos fora da fala ficam intactos.
    pub duck_gain: f32,
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
            detect_gender: true,
            align_to_voice: true,
            match_voice: true,
            background_gain: 0.9,
            duck_gain: 0.2,
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
    /// Opcional: o mesmo áudio em alta qualidade (estéreo, 44,1 kHz), para a
    /// separação voz/instrumental e para o fundo da mixagem.
    pub audio_hq: Option<PathBuf>,
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
    /// Legenda com o texto reconhecido no idioma original.
    pub srt_source: PathBuf,
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
                report(Stage::Separate, 0, 100, "separando voz e fundo");
                let src = job.audio_hq.as_ref().unwrap_or(&job.audio);
                let on_progress: Arc<dyn Fn(u8) + Send + Sync> = {
                    let progress = progress.clone();
                    Arc::new(move |pct: u8| {
                        if let Some(p) = &progress {
                            let _ = p.send(Progress {
                                stage: Stage::Separate,
                                done: pct.min(100) as usize,
                                total: 100,
                                message: format!("separando voz e fundo {pct}%"),
                            });
                        }
                    })
                };
                let (v, b) = sep.separate_with_progress(src, &job.work_dir, on_progress)?;
                report(Stage::Separate, 100, 100, "separação concluída");
                (v, Some(b))
            }
            _ => {
                // Sem separação: o original inteiro é o fundo (com ducking na
                // mixagem). Perder a música/ruído seria pior do que ouvi-los
                // baixinho sob a dublagem.
                (job.audio.clone(), Some(job.audio.clone()))
            }
        };
        let ducked = self.engines.separator.is_none() || !self.config.separate;
        self.check_cancel()?;

        // 2. Transcrição -------------------------------------------------------
        report(Stage::Transcribe, 0, 100, "transcrevendo");
        let on_progress: Arc<dyn Fn(u8) + Send + Sync> = {
            let progress = progress.clone();
            Arc::new(move |pct: u8| {
                if let Some(p) = &progress {
                    let _ = p.send(Progress {
                        stage: Stage::Transcribe,
                        done: pct.min(100) as usize,
                        total: 100,
                        message: format!("transcrevendo {pct}%"),
                    });
                }
            })
        };
        let mut segments =
            self.engines.transcriber.transcribe_with_progress(&voice_path, self.config.source_lang.as_ref(), on_progress)?;
        segments.retain(|s| !s.source_text.trim().is_empty() && s.end > s.start);
        for (i, s) in segments.iter_mut().enumerate() {
            s.index = i;
        }
        report(Stage::Transcribe, 100, 100, &format!("{} segmentos", segments.len()));
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

        // 3a. Limpeza e alinhamento pela voz isolada ----------------------------
        let before = segments.len();
        segments.retain(|s| !is_ghost(s));
        if segments.len() != before {
            info!(removidos = before - segments.len(), "trechos fantasma descartados");
        }
        if segments.is_empty() {
            return Err(DubError::Audio("nenhuma fala detectada".into()));
        }
        let separated = background_path.is_some() && self.engines.separator.is_some() && self.config.separate;
        if separated && self.config.align_to_voice {
            if let Ok(voice) = AudioBuffer::read_wav(&voice_path) {
                let moved = snap_to_voice(&mut segments, &voice);
                info!(ajustados = moved, "início/fim das frases alinhados à voz original");
            }
        }

        // 3b. Gênero da voz original ------------------------------------------
        if self.config.detect_gender {
            match AudioBuffer::read_wav(&voice_path) {
                Ok(voice) => {
                    estimate_genders(&mut segments, &voice);
                    let (m, f) = segments.iter().fold((0, 0), |(m, f), s| match s.gender {
                        Some(crate::Gender::Male) => (m + 1, f),
                        Some(crate::Gender::Female) => (m, f + 1),
                        None => (m, f),
                    });
                    info!(masculinos = m, femininos = f, "gênero estimado por trecho");
                }
                Err(e) => warn!("sem estimativa de gênero: {e}"),
            }
        }
        self.check_cancel()?;

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
        let output = self.fit_and_mix(segments, background_path, ducked, &job.work_dir, &report)?;
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
        let stretcher_s = self.stretcher.clone();
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
                        let speed_hint = if pass_speed { estimate_speed(&seg.source_text, &text, &cfg_s.timing) } else { 1.0 };
                        let mut audio =
                            synthesizer.synthesize_segment(&seg, &text, &cfg_s.target_lang, ref_path.as_deref(), speed_hint)?;
                        if cfg_s.match_voice {
                            match_voice(&mut audio, &seg, stretcher_s.as_ref())?;
                        }
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
        ducked: bool,
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
            let a =
                if (plan.speed - 1.0).abs() > 0.01 { self.stretcher.stretch(&dubbed[i], plan.speed)? } else { dubbed[i].clone() };
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
        // O fundo mantém os canais originais (estéreo fica estéreo); a voz
        // dublada entra no centro, igual em todos os canais.
        report(Stage::Mix, 0, 1, "mixando");
        let mixed_wav = work_dir.join("dub_mix.wav");
        match &background {
            Some(bg) => {
                let (bg_sr, bg_chans) = AudioBuffer::read_wav_channels(bg)?;
                let windows: Vec<(f64, f64)> = segments.iter().map(|s| (s.start, s.end)).collect();
                let mut out: Vec<Vec<f32>> = Vec::with_capacity(bg_chans.len().max(1));
                for ch in bg_chans {
                    let mut bgb = resample_if_needed(AudioBuffer::new(bg_sr, ch), sr);
                    if ducked {
                        bgb.duck(&windows, self.config.duck_gain, 0.08);
                    }
                    let mut mixed = voice_track.clone();
                    mixed.mix_at(&bgb, 0.0, self.config.background_gain)?;
                    out.push(mixed.samples);
                }
                // Limitador conjunto, para não deslocar a imagem estéreo.
                let peak = out.iter().flat_map(|c| c.iter()).fold(0.0f32, |m, v| m.max(v.abs()));
                if peak > 0.95 {
                    let g = 0.95 / peak;
                    for c in &mut out {
                        for v in c.iter_mut() {
                            *v *= g;
                        }
                    }
                }
                AudioBuffer::write_wav_channels(&mixed_wav, sr, &out)?;
            }
            None => {
                let mut mixed = voice_track.clone();
                mixed.limit_peak(0.95);
                mixed.write_wav(&mixed_wav)?;
            }
        }

        let srt = work_dir.join("dub.srt");
        std::fs::write(&srt, to_srt(&segments))?;
        // Também o texto reconhecido no idioma original, para conferir onde um
        // erro nasceu (ouvido ou tradução).
        let srt_source = work_dir.join("dub.src.srt");
        std::fs::write(&srt_source, to_srt_source(&segments))?;
        report(Stage::Mix, 1, 1, "");

        Ok(DubOutput { mixed_wav, voice_wav, background_wav: background, segments, srt, srt_source })
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
/// Trecho "fantasma": o whisper às vezes devolve um "E-" ou um "." de 2 s em
/// cima de música. Sem letras suficientes ou curto demais, não vira dublagem.
fn is_ghost(s: &Segment) -> bool {
    let letters = s.source_text.chars().filter(|c| c.is_alphanumeric()).count();
    letters < 3 || (s.end - s.start) < 0.25
}

/// Ajusta `start`/`end` de cada trecho à energia da voz isolada: o início vai
/// para o primeiro quadro com voz perto da marca do whisper, o fim para o
/// último. Nunca invade o trecho vizinho. Devolve quantos trechos mudaram.
pub fn snap_to_voice(segments: &mut [Segment], voice: &AudioBuffer) -> usize {
    let sr = voice.sample_rate as f64;
    let frame = 0.02; // 20 ms
    let n = (frame * sr) as usize;
    if n == 0 || voice.samples.len() < n * 10 {
        return 0;
    }
    let frames = voice.samples.len() / n;
    let rms: Vec<f32> = (0..frames)
        .map(|i| {
            let x = &voice.samples[i * n..(i + 1) * n];
            (x.iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt()
        })
        .collect();
    let t_of = |i: usize| i as f64 * frame;
    let idx_of = |t: f64| ((t / frame).floor().max(0.0) as usize).min(frames.saturating_sub(1));
    let mut moved = 0;
    let len = segments.len();
    for i in 0..len {
        let (start, end) = (segments[i].start, segments[i].end);
        let prev_end = if i > 0 { segments[i - 1].end } else { 0.0 };
        let next_start = segments.get(i + 1).map(|s| s.start).unwrap_or(f64::MAX);
        // Limiar relativo ao pico de voz na janela do trecho (com folga).
        let (a, b) = (idx_of((start - 0.6).max(prev_end)), idx_of((end + 0.6).min(next_start)));
        if b <= a + 2 {
            continue;
        }
        let peak = rms[a..=b].iter().cloned().fold(0.0f32, f32::max);
        if peak < 0.02 {
            continue; // sem voz audível: deixa como está
        }
        let thr = (peak * 0.15).max(0.012);
        // Início: primeiro quadro acima do limiar entre (start − 0,6 s) e (start + 1,2 s).
        let s_hi = idx_of((start + 1.2).min(end));
        let new_start = (a..=s_hi).find(|&k| rms[k] > thr).map(t_of).unwrap_or(start);
        // Fim: último quadro acima do limiar entre (end − 1,5 s) e (end + 0,6 s).
        let e_lo = idx_of((end - 1.5).max(new_start + 0.3));
        let new_end = (e_lo..=b).rev().find(|&k| rms[k] > thr).map(|k| t_of(k + 1)).unwrap_or(end);
        let new_start = new_start.max(prev_end);
        let new_end = new_end.min(next_start).max(new_start + 0.3);
        if (new_start - start).abs() > 0.05 || (new_end - end).abs() > 0.05 {
            moved += 1;
        }
        segments[i].start = new_start;
        segments[i].end = new_end;
    }
    moved
}

/// Agrupa trechos em "falantes" pelo tom: ordena os f0 e corta onde dois
/// vizinhos distam mais de 20%. Trechos sem f0 vão para o grupo mais próximo
/// no tempo. Só roda quando não há diarização.
fn cluster_by_pitch(segments: &mut [Segment], f0s: &[Option<f32>]) {
    let mut known: Vec<(usize, f32)> = f0s.iter().enumerate().filter_map(|(i, f)| f.map(|v| (i, v))).collect();
    if known.len() < 2 {
        for s in segments.iter_mut() {
            s.speaker = Some(Speaker("voz-1".into()));
        }
        return;
    }
    known.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut cluster_of = vec![0usize; segments.len()];
    let mut c = 0usize;
    let mut assigned = vec![false; segments.len()];
    cluster_of[known[0].0] = 0;
    assigned[known[0].0] = true;
    for w in known.windows(2) {
        if w[1].1 / w[0].1 > 1.2 {
            c += 1;
        }
        cluster_of[w[1].0] = c;
        assigned[w[1].0] = true;
    }
    // Sem f0: herda do trecho anterior (ou do próximo).
    let mut last = cluster_of[known[0].0];
    for i in 0..segments.len() {
        if assigned[i] {
            last = cluster_of[i];
        } else {
            cluster_of[i] = last;
        }
    }
    for (s, k) in segments.iter_mut().zip(&cluster_of) {
        s.speaker = Some(Speaker(format!("voz-{}", k + 1)));
    }
}

/// Aproxima a dublagem da voz original do trecho: desloca o tom até o f0
/// original (±4 semitons) e inclina o espectro na direção do brilho original.
pub fn match_voice(audio: &mut AudioBuffer, seg: &Segment, stretcher: &dyn Stretcher) -> Result<()> {
    if let (Some(target), Some(current)) = (seg.f0_hz, audio.median_f0()) {
        let ratio = (target / current).clamp(0.794, 1.26); // ±4 semitons
        if (ratio - 1.0).abs() > 0.03 {
            *audio = audio.pitch_shifted(ratio, stretcher)?;
        }
    }
    if let (Some(target), Some(current)) = (seg.brightness, audio.brightness()) {
        if current > 1e-4 {
            let tilt = ((target / current).ln() * 0.5).clamp(-0.5, 0.5);
            audio.tilt(tilt);
        }
    }
    Ok(())
}

/// Estima o gênero de cada segmento pelo f0 do trecho na voz isolada. Com
/// diarização, cada falante recebe o voto majoritário (ponderado pela
/// duração) dos seus trechos, para a voz não trocar no meio de uma fala.
pub fn estimate_genders(segments: &mut [Segment], voice: &AudioBuffer) {
    use std::collections::HashMap;
    let mut per_seg: Vec<Option<f32>> = Vec::with_capacity(segments.len());
    for s in segments.iter_mut() {
        let piece = voice.slice_seconds(s.start, s.end);
        let f0 = piece.median_f0();
        s.f0_hz = f0;
        s.brightness = piece.brightness();
        per_seg.push(f0);
    }
    // Sem diarização, agrupa os trechos por faixa de tom: vozes diferentes
    // costumam ficar a mais de ~20% uma da outra. Dá "falantes" aproximados
    // para a voz da dublagem não ser a mesma para todo mundo.
    if segments.iter().all(|s| s.speaker.is_none()) {
        cluster_by_pitch(segments, &per_seg);
    }
    // Voto por falante: soma de durações por gênero.
    let mut votes: HashMap<String, (f64, f64)> = HashMap::new(); // (masc, fem)
    for (s, f0) in segments.iter().zip(&per_seg) {
        if let (Some(spk), Some(f0)) = (&s.speaker, f0) {
            let e = votes.entry(spk.0.clone()).or_default();
            match crate::Gender::from_f0(*f0) {
                crate::Gender::Male => e.0 += s.end - s.start,
                crate::Gender::Female => e.1 += s.end - s.start,
            }
        }
    }
    // Sem diarização: voto global, usado só onde o trecho não deu f0.
    let global = {
        let (m, f) = segments.iter().zip(&per_seg).fold((0.0, 0.0), |(m, f), (s, f0)| match f0.map(crate::Gender::from_f0) {
            Some(crate::Gender::Male) => (m + (s.end - s.start), f),
            Some(crate::Gender::Female) => (m, f + (s.end - s.start)),
            None => (m, f),
        });
        if m + f > 0.0 {
            Some(if f > m { crate::Gender::Female } else { crate::Gender::Male })
        } else {
            None
        }
    };
    // Tom por falante (mediana), para a voz não oscilar trecho a trecho.
    let mut spk_f0: HashMap<String, Vec<f32>> = HashMap::new();
    for (s, f0) in segments.iter().zip(&per_seg) {
        if let (Some(spk), Some(f0)) = (&s.speaker, f0) {
            spk_f0.entry(spk.0.clone()).or_default().push(*f0);
        }
    }
    let spk_median: HashMap<String, f32> = spk_f0
        .into_iter()
        .map(|(k, mut v)| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (k, v[v.len() / 2])
        })
        .collect();
    for s in segments.iter_mut() {
        if let Some(spk) = &s.speaker {
            if let Some(m) = spk_median.get(&spk.0) {
                s.f0_hz = Some(*m);
            }
        }
    }
    for (s, f0) in segments.iter_mut().zip(&per_seg) {
        s.gender = match (&s.speaker, f0) {
            (Some(spk), _) if votes.get(&spk.0).map(|(m, f)| m + f > 0.0).unwrap_or(false) => {
                let (m, f) = votes[&spk.0];
                Some(if f > m { crate::Gender::Female } else { crate::Gender::Male })
            }
            (_, Some(f0)) => Some(crate::Gender::from_f0(*f0)),
            _ => global,
        };
    }
}

fn resample_if_needed(a: AudioBuffer, target_sr: u32) -> AudioBuffer {
    if a.sample_rate == target_sr {
        return a;
    }
    a.resampled(target_sr)
}

fn srt_time(t: f64) -> String {
    let ms = (t * 1000.0).round() as u64;
    format!("{:02}:{:02}:{:02},{:03}", ms / 3_600_000, (ms / 60_000) % 60, (ms / 1000) % 60, ms % 1000)
}

pub fn to_srt(segments: &[Segment]) -> String {
    srt_with(segments, |s| s.target_text.as_deref().unwrap_or(&s.source_text))
}

/// SRT com o texto reconhecido no idioma original.
pub fn to_srt_source(segments: &[Segment]) -> String {
    srt_with(segments, |s| s.source_text.as_str())
}

fn srt_with<'a>(segments: &'a [Segment], text: impl Fn(&'a Segment) -> &'a str) -> String {
    let mut out = String::new();
    for (i, s) in segments.iter().enumerate() {
        out.push_str(&format!("{}\n{} --> {}\n{}\n\n", i + 1, srt_time(s.start), srt_time(s.end), text(s)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snap_moves_segment_to_voice_energy() {
        // Voz (ruído) só entre 1,0 s e 2,0 s; whisper marcou 0,5–2,8.
        let sr = 16_000u32;
        let mut samples = vec![0.0f32; sr as usize * 4];
        let mut seed = 7u32;
        for v in samples.iter_mut().take(sr as usize * 2).skip(sr as usize) {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v = ((seed >> 9) as f32 / (1u32 << 23) as f32 - 1.0) * 0.3;
        }
        let voice = AudioBuffer::new(sr, samples);
        let mut segs = vec![Segment::new(0, 0.5, 2.8, "hello world")];
        let moved = snap_to_voice(&mut segs, &voice);
        assert_eq!(moved, 1);
        assert!((segs[0].start - 1.0).abs() < 0.06, "start {}", segs[0].start);
        assert!((segs[0].end - 2.0).abs() < 0.06, "end {}", segs[0].end);
    }

    #[test]
    fn ghosts_are_filtered() {
        assert!(is_ghost(&Segment::new(0, 0.0, 2.0, "E-")));
        assert!(is_ghost(&Segment::new(0, 0.0, 0.1, "hello")));
        assert!(!is_ghost(&Segment::new(0, 0.0, 1.0, "hello")));
    }
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
        let engines =
            EngineSet { transcriber: f.clone(), translator: f.clone(), synthesizer: f.clone(), separator: None, diarizer: None };
        let cfg = PipelineConfig { sample_rate: 16000, ..Default::default() };
        let (tx, rx) = crossbeam_channel::unbounded();
        let out =
            Pipeline::new(engines, cfg).run(&DubJob { audio: src, audio_hq: None, work_dir: dir.clone() }, Some(tx)).unwrap();

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
