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
use crate::audio::{rms_of, soft_limit_channels};
use crate::engines::EngineSet;
use crate::error::{DubError, Result};
use crate::segment::{clean_caption_text, merge_fragments, split_long, Lang, Segment, SegmentKind, Speaker};
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

/// O que fazer com trechos cantados.
///
/// Canto dublado por TTS vira um narrador falando por cima da música — nunca
/// soa bem. Em `Auto`, o pipeline mede, com a voz e o fundo separados, quais
/// trechos têm música tão alta quanto a voz; esses são tratados como canto:
/// a voz original fica na mix e a tradução vai só para a legenda. Fala comum
/// (introdução, diálogo no meio do clipe) continua dublada.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ContentMode {
    /// Decide por trecho: mantém o canto original e dubla a fala.
    Auto,
    /// Dubla todos os trechos, inclusive canto (padrão).
    #[default]
    Speech,
    /// É música: mantém o canto original e dubla só o que for claramente falado.
    Music,
}

impl std::fmt::Display for ContentMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ContentMode::Auto => "auto",
            ContentMode::Speech => "speech",
            ContentMode::Music => "music",
        })
    }
}

impl std::str::FromStr for ContentMode {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(ContentMode::Auto),
            "speech" | "fala" => Ok(ContentMode::Speech),
            "music" | "musica" | "música" => Ok(ContentMode::Music),
            other => Err(format!("modo de conteúdo desconhecido: {other} (auto | speech | music)")),
        }
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
    /// Com separação, o instrumental é abaixado para este volume só enquanto a
    /// voz dublada toca (*sidechain*); fora da fala fica em `background_gain`.
    /// 1.0 desliga.
    #[serde(default = "default_music_duck_gain")]
    pub music_duck_gain: f32,
    /// Volume da voz dublada (0..1).
    pub voice_gain: f32,
    /// Quantos dB a voz dublada fica **acima** do fundo (música/efeitos) nos
    /// trechos em que ela toca. Fala sobre música só é entendida com uns 6 dB
    /// de folga; com 0 ou menos, a dublagem "some" dentro da música.
    #[serde(default = "default_voice_over_music_db")]
    pub voice_over_music_db: f32,
    /// Tratar a voz dublada antes da mixagem: passa-altas em 110 Hz, presença
    /// (+3 dB em 3 kHz) e compressor — voz clara e estável sobre a música.
    #[serde(default = "default_true")]
    pub voice_polish: bool,
    /// Em trechos sobrepostos, reduz a voz dublada por este fator (fica mais
    /// natural do que duas dublagens brigando).
    pub overlap_gain: f32,
    /// Fala, música ou decidir por trecho.
    #[serde(default)]
    pub content: ContentMode,
    /// Ajustar o volume final para o mesmo nível (RMS) do áudio original, em
    /// vez de deixar a mix mais baixa que a faixa original do vídeo.
    #[serde(default = "default_true")]
    pub match_loudness: bool,
    /// Trechos mais longos que isto são quebrados antes da tradução/TTS.
    #[serde(default = "default_max_segment_secs")]
    pub max_segment_secs: f64,
    pub timing: TimingConfig,
    /// Sample rate da trilha final.
    pub sample_rate: u32,
}

fn default_music_duck_gain() -> f32 {
    0.5
}
fn default_voice_over_music_db() -> f32 {
    6.0
}
fn default_true() -> bool {
    true
}
fn default_max_segment_secs() -> f64 {
    8.0
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
            music_duck_gain: 0.5,
            voice_gain: 1.0,
            voice_over_music_db: default_voice_over_music_db(),
            voice_polish: true,
            overlap_gain: 0.6,
            content: ContentMode::Speech,
            match_loudness: true,
            max_segment_secs: 8.0,
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
        // Marcações de legenda ([Music], >>, ♪…) não são fala: fora antes de
        // traduzir, senão o TTS lê "música" em voz alta.
        for s in segments.iter_mut() {
            s.source_text = clean_caption_text(&s.source_text);
        }
        segments.retain(|s| !s.source_text.trim().is_empty() && s.end > s.start);
        // Frases cortadas no meio pela legenda viram uma só antes da tradução.
        let before_merge = segments.len();
        segments = merge_fragments(segments, self.config.max_segment_secs.max(2.0));
        if segments.len() != before_merge {
            info!(antes = before_merge, depois = segments.len(), "pedaços de frase juntados");
        }
        // Vale para qualquer transcritor (whisper em Python, legenda pronta…):
        // blocos de 15–20 s não têm como ser encaixados nem dublados direito.
        let before_split = segments.len();
        segments = split_long(segments, self.config.max_segment_secs.max(2.0));
        if segments.len() != before_split {
            info!(antes = before_split, depois = segments.len(), "trechos longos quebrados");
        }
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

        // 3a'. Fala ou canto? ---------------------------------------------------
        let sung = match (self.config.content, separated, &background_path) {
            (ContentMode::Speech, _, _) => 0,
            (mode, true, Some(bg)) => match (AudioBuffer::read_wav(&voice_path), AudioBuffer::read_wav(bg)) {
                (Ok(voice), Ok(background)) => classify_singing(&mut segments, &voice, &background, mode),
                _ => 0,
            },
            // Sem separação não dá para medir; em modo música, assume que é
            // tudo canto (fica legenda + áudio original).
            (ContentMode::Music, _, _) => {
                segments.iter_mut().for_each(|s| s.kind = SegmentKind::Singing);
                segments.len()
            }
            _ => 0,
        };
        if sung > 0 {
            info!(cantados = sung, total = segments.len(), "trechos cantados: voz original mantida, tradução só na legenda");
            report(
                Stage::Transcribe,
                100,
                100,
                &format!("{sung} de {} trechos são canto (voz original mantida)", segments.len()),
            );
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
        let original_voice = if separated && segments.iter().any(Segment::is_singing) { Some(voice_path.clone()) } else { None };
        let reference = job.audio_hq.clone().unwrap_or_else(|| job.audio.clone());
        let output = self.fit_and_mix(segments, background_path, original_voice, &reference, ducked, &job.work_dir, &report)?;
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
                    s.target_text = Some(clean_caption_text(&t));
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
                    // Canto: a tradução fica só na legenda; a voz original volta na mix.
                    if seg.is_singing() {
                        done += 1;
                        if let Some(p) = &progress_s {
                            let _ = p.send(Progress { stage: Stage::Synthesize, done, total, message: String::new() });
                        }
                        tx_done.send(Ok(seg)).map_err(|_| DubError::ChannelClosed("done"))?;
                        continue;
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
                        // Silêncio no começo do áudio do TTS vira atraso em relação à fala original.
                        audio.trim_silence(0.01, 0.02);
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

    #[allow(clippy::too_many_arguments)]
    fn fit_and_mix(
        &self,
        mut segments: Vec<Segment>,
        background: Option<PathBuf>,
        original_voice: Option<PathBuf>,
        reference: &Path,
        ducked: bool,
        work_dir: &Path,
        report: &dyn Fn(Stage, usize, usize, &str),
    ) -> Result<DubOutput> {
        let sr = self.config.sample_rate;
        let total = segments.len();

        // Carrega as dublagens e mede (trechos cantados não têm dublagem).
        report(Stage::Fit, 0, total, "ajustando tempo");
        let mut dubbed: Vec<Option<AudioBuffer>> = Vec::with_capacity(total);
        for s in &segments {
            if s.is_singing() {
                dubbed.push(None);
                continue;
            }
            let p = s.dubbed_wav.as_ref().ok_or_else(|| DubError::Audio(format!("segmento {} sem áudio", s.index)))?;
            let a = AudioBuffer::read_wav(p)?;
            dubbed.push(Some(resample_if_needed(a, sr)));
        }
        let lens: Vec<f64> = dubbed.iter().map(|a| a.as_ref().map(|a| a.duration()).unwrap_or(0.0)).collect();
        let plans = plan_all(&mut segments, &lens, &self.config.timing);

        // Estica e posiciona.
        let total_len = background
            .as_ref()
            .and_then(|b| AudioBuffer::read_wav(b).ok())
            .map(|b| b.duration())
            .unwrap_or_else(|| segments.last().map(|s| s.end + 2.0).unwrap_or(0.0));
        let mut voice_track = AudioBuffer::silence(sr, total_len);
        // Janelas em que a dublagem realmente soa (com o transbordo), para o
        // ducking seguir a voz e não a marcação do whisper.
        let mut speech_windows: Vec<(f64, f64)> = Vec::new();
        let mut sung_windows: Vec<(f64, f64)> = Vec::new();

        for (i, (seg, plan)) in segments.iter().zip(&plans).enumerate() {
            self.check_cancel()?;
            let Some(src) = &dubbed[i] else {
                sung_windows.push((seg.start, seg.end));
                info!(
                    index = seg.index,
                    inicio = format!("{:.2}", seg.start),
                    dur = format!("{:.2}", seg.duration()),
                    "canto: voz original mantida"
                );
                report(Stage::Fit, i + 1, total, "");
                continue;
            };
            let a = if (plan.speed - 1.0).abs() > 0.01 { self.stretcher.stretch(src, plan.speed)? } else { src.clone() };
            if plan.overflow > 0.3 {
                warn!(index = seg.index, overflow = plan.overflow, "dublagem transborda além da janela");
            }
            let gain = if seg.overlapped { self.config.voice_gain * self.config.overlap_gain } else { self.config.voice_gain };
            voice_track.mix_at(&a, seg.start, gain)?;
            speech_windows.push((seg.start, seg.start + a.duration()));
            // Um log por trecho: é por aqui que se descobre por que um cue saiu
            // mudo, corrido ou atrasado.
            info!(
                index = seg.index,
                inicio = format!("{:.2}", seg.start),
                janela = format!("{:.2}", seg.duration()),
                tts = format!("{:.2}", src.duration()),
                final_ = format!("{:.2}", a.duration()),
                speed = format!("{:.2}", plan.speed),
                transborda = format!("{:.2}", plan.overflow),
                chars = seg.target_text.as_deref().map(|t| t.chars().count()).unwrap_or(0),
                "trecho posicionado"
            );
            debug!(index = seg.index, speed = plan.speed, "segmento posicionado");
            report(Stage::Fit, i + 1, total, "");
        }

        // Tratamento da voz: tira o grave que embola com o baixo da música,
        // realça a presença (3 kHz) e estabiliza o volume com um compressor.
        if self.config.voice_polish && !speech_windows.is_empty() {
            voice_track.highpass(110.0);
            voice_track.peaking_eq(3000.0, 1.0, 3.0);
            voice_track.compress(-24.0, 3.0, 5.0, 120.0);
        }

        let voice_wav = work_dir.join("voice_dubbed.wav");
        {
            // Só a voz, para conferência/edição: pico normalizado num cópia, para
            // não mexer no nível relativo usado na mixagem.
            let mut v = voice_track.clone();
            v.limit_peak(0.95);
            v.write_wav(&voice_wav)?;
        }

        // Mixagem ----------------------------------------------------------
        // O fundo mantém os canais originais (estéreo fica estéreo); a voz
        // dublada entra no centro, igual em todos os canais.
        report(Stage::Mix, 0, 1, "mixando");
        let mixed_wav = work_dir.join("dub_mix.wav");
        // Voz original (isolada) de volta nos trechos cantados.
        let original_voice_buf = match (&original_voice, sung_windows.is_empty()) {
            (Some(p), false) => Some(resample_if_needed(AudioBuffer::read_wav(p)?, sr)),
            _ => None,
        };
        // Fundo já com ducking e ganho, por canal.
        let backgrounds: Vec<AudioBuffer> = match &background {
            Some(bg) => {
                let (bg_sr, bg_chans) = AudioBuffer::read_wav_channels(bg)?;
                bg_chans
                    .into_iter()
                    .map(|ch| {
                        let mut bgb = resample_if_needed(AudioBuffer::new(bg_sr, ch), sr);
                        if ducked {
                            // Sem separação: o original inteiro (com a voz) abaixa bem sob a dublagem.
                            bgb.duck(&speech_windows, self.config.duck_gain, 0.08);
                        } else if self.config.music_duck_gain < 0.999 {
                            // Com separação: o instrumental cede espaço só enquanto a dublagem toca.
                            bgb.duck(&speech_windows, self.config.music_duck_gain, 0.15);
                        }
                        bgb.scale(self.config.background_gain);
                        bgb
                    })
                    .collect()
            }
            None => Vec::new(),
        };

        // Voz × música: a voz dublada precisa ficar `voice_over_music_db` acima
        // do fundo enquanto toca, senão "some" dentro da música.
        let voice_rms = voice_track.rms_in_windows(&speech_windows);
        let bg_rms = backgrounds.iter().map(|b| b.rms_in_windows(&speech_windows)).fold(0.0f32, f32::max);
        if voice_rms > 1e-4 && bg_rms > 1e-4 {
            let before = db(voice_rms) - db(bg_rms);
            let target = self.config.voice_over_music_db;
            let g = (10f32.powf((target - before) / 20.0)).clamp(0.25, 8.0);
            voice_track.scale(g);
            let after = before + db(g);
            info!(
                antes_db = format!("{before:+.1}"),
                depois_db = format!("{after:+.1}"),
                alvo_db = format!("{target:+.1}"),
                "voz × fundo nos trechos dublados"
            );
            report(Stage::Mix, 0, 1, &format!("voz {after:+.1} dB acima do fundo (era {before:+.1} dB)"));
        }

        let mut out: Vec<Vec<f32>> = if backgrounds.is_empty() {
            vec![voice_track.samples.clone()]
        } else {
            let mut out: Vec<Vec<f32>> = Vec::with_capacity(backgrounds.len());
            for bgb in &backgrounds {
                let mut mixed = voice_track.clone();
                mixed.mix_at(bgb, 0.0, 1.0)?;
                if let Some(ov) = &original_voice_buf {
                    mixed.mix_windows(ov, &sung_windows, 1.0, 0.05)?;
                }
                out.push(mixed.samples);
            }
            out
        };

        // Volume final: igual ao do original, em vez de "o que sobrou" depois
        // de tirar a voz e escalar pelo pico. Um estouro isolado passa pelo
        // limitador suave sem abaixar o vídeo inteiro.
        if self.config.match_loudness {
            if let Ok((ref_sr, ref_chans)) = AudioBuffer::read_wav_channels(reference) {
                let _ = ref_sr;
                let ref_rms = ref_chans.iter().map(|c| rms_of(c)).fold(0.0f32, f32::max);
                let mix_rms = out.iter().map(|c| rms_of(c)).fold(0.0f32, f32::max);
                if ref_rms > 1e-4 && mix_rms > 1e-4 {
                    let g = (ref_rms / mix_rms).clamp(0.5, 4.0);
                    info!(
                        original_db = format!("{:.1}", db(ref_rms)),
                        mix_db = format!("{:.1}", db(mix_rms)),
                        ganho = format!("{g:.2}"),
                        "volume casado com o original"
                    );
                    for c in &mut out {
                        for v in c.iter_mut() {
                            *v *= g;
                        }
                    }
                }
            }
        }
        // Teto em −1 dBFS (0,89): sem estouro nem distorção ao converter para AAC.
        soft_limit_channels(&mut out, 0.75, 0.89);
        if out.len() == 1 {
            AudioBuffer::new(sr, std::mem::take(&mut out[0])).write_wav(&mixed_wav)?;
        } else {
            AudioBuffer::write_wav_channels(&mixed_wav, sr, &out)?;
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

fn db(x: f32) -> f32 {
    20.0 * (x.max(1e-9)).log10()
}

/// Marca como canto os trechos em que a música (fundo separado) é tão alta
/// quanto a voz. Devolve quantos foram marcados.
///
/// Por trecho mede `fundo_rms / voz_rms`. Em [`ContentMode::Auto`], um trecho
/// é canto se a relação passa de 0,8 **e** o vídeo como um todo é musical
/// (pelo menos 40% do tempo falado com música alta) — uma entrevista com
/// trilha de fundo num trecho só não vira "música". Em [`ContentMode::Music`]
/// basta a relação passar de 0,35: só fala claramente a seco é dublada.
pub fn classify_singing(segments: &mut [Segment], voice: &AudioBuffer, background: &AudioBuffer, mode: ContentMode) -> usize {
    if segments.is_empty() || mode == ContentMode::Speech {
        return 0;
    }
    let ratios: Vec<f32> = segments
        .iter()
        .map(|s| {
            let v = voice.slice_seconds(s.start, s.end).rms();
            let b = background.slice_seconds(s.start, s.end).rms();
            if v < 1e-4 {
                // Voz isolada praticamente muda: não há o que dublar nem manter.
                return 0.0;
            }
            b / v
        })
        .collect();
    let threshold = match mode {
        ContentMode::Music => 0.35,
        _ => 0.8,
    };
    if mode == ContentMode::Auto {
        let total: f64 = segments.iter().map(Segment::duration).sum::<f64>().max(1e-3);
        let musical: f64 = segments.iter().zip(&ratios).filter(|(_, r)| **r > threshold).map(|(s, _)| s.duration()).sum();
        if musical / total < 0.4 {
            debug!(fracao = musical / total, "vídeo tratado como fala");
            return 0;
        }
        // Música alta não basta: em jogo e filme a trilha e os efeitos cobrem o
        // diálogo o tempo todo. Só é "vídeo musical" se a própria voz canta —
        // notas sustentadas em boa parte do tempo com voz.
        let sung: f64 = segments
            .iter()
            .filter(|s| voice.slice_seconds(s.start, s.end).sustained_ratio().map(|r| r >= 0.12).unwrap_or(false))
            .map(Segment::duration)
            .sum();
        if sung / total < 0.35 {
            info!(
                fracao_cantada = format!("{:.2}", sung / total),
                "fundo alto, mas a voz é falada (jogo/filme): todos os trechos serão dublados"
            );
            return 0;
        }
    }
    let mut n = 0;
    for (s, r) in segments.iter_mut().zip(&ratios) {
        if *r > threshold {
            s.kind = SegmentKind::Singing;
            n += 1;
        } else {
            s.kind = SegmentKind::Speech;
        }
    }
    n
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

/// Sem diarização, cria no máximo dois "falantes": voz masculina e voz
/// feminina. Agrupar por faixas de tom mais finas fazia um único personagem
/// (um narrador que sobe e desce o tom) sair com duas vozes diferentes.
/// O gênero minoritário só vira um falante próprio se tiver pelo menos 3
/// trechos e 20% do tempo com tom medido; abaixo disso é ruído de medição
/// (música por baixo, grito, sussurro) e tudo fica com a voz majoritária.
fn cluster_by_pitch(segments: &mut [Segment], f0s: &[Option<f32>]) {
    let mut dur = [0.0f64; 2]; // [masc, fem]
    let mut count = [0usize; 2];
    for (s, f) in segments.iter().zip(f0s) {
        if let Some(f) = f {
            let k = if crate::Gender::from_f0(*f) == crate::Gender::Female { 1 } else { 0 };
            dur[k] += s.duration();
            count[k] += 1;
        }
    }
    let total = (dur[0] + dur[1]).max(1e-6);
    let major = if dur[1] > dur[0] { 1 } else { 0 };
    let minor = 1 - major;
    // Além disso os dois grupos precisam ser mesmo duas vozes: um narrador em
    // torno de 155 Hz fica metade de cada lado do corte de 165 Hz, mas as
    // medianas dos dois lados ficam coladas. Homem e mulher distam ≥ 30%.
    let median_of = |k: usize| -> f32 {
        let mut v: Vec<f32> =
            f0s.iter().flatten().copied().filter(|f| (crate::Gender::from_f0(*f) == crate::Gender::Female) == (k == 1)).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if v.is_empty() {
            0.0
        } else {
            v[v.len() / 2]
        }
    };
    let apart = median_of(0) > 0.0 && median_of(1) / median_of(0) >= 1.3;
    let two_voices = count[minor] >= 3 && dur[minor] / total >= 0.2 && apart;
    let name = |k: usize| if k == 1 { "voz-feminina" } else { "voz-masculina" };
    let mut last = major;
    for (s, f) in segments.iter_mut().zip(f0s) {
        let k = match f {
            Some(f) if two_voices => {
                if crate::Gender::from_f0(*f) == crate::Gender::Female {
                    1
                } else {
                    0
                }
            }
            Some(_) => major,
            None => last,
        };
        last = k;
        s.speaker = Some(Speaker(name(k).into()));
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
    // Sem diarização: no máximo uma voz masculina e uma feminina.
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
    fn singing_detected_where_music_is_loud() {
        let sr = 16_000u32;
        let tone = |secs: f64, amp: f32| {
            AudioBuffer::new(sr, (0..(sr as f64 * secs) as usize).map(|i| ((i as f32) * 0.07).sin() * amp).collect())
        };
        // Voz constante; fundo: silêncio nos 2 primeiros segundos (fala a seco), depois alto.
        let voice = tone(8.0, 0.3);
        let mut bg = tone(8.0, 0.3);
        for v in bg.samples.iter_mut().take(sr as usize * 2) {
            *v = 0.0;
        }
        let mut segs = vec![
            Segment::new(0, 0.0, 2.0, "spoken intro"),
            Segment::new(1, 2.0, 5.0, "sung line"),
            Segment::new(2, 5.0, 8.0, "another sung line"),
        ];
        let n = classify_singing(&mut segs, &voice, &bg, ContentMode::Auto);
        assert_eq!(n, 2);
        assert_eq!(segs[0].kind, SegmentKind::Speech);
        assert!(segs[1].is_singing() && segs[2].is_singing());
        // Modo fala: nunca marca.
        let mut segs2 = segs.clone();
        assert_eq!(classify_singing(&mut segs2, &voice, &bg, ContentMode::Speech), 0);
        // Auto com pouco tempo musical: tratado como fala.
        let mut segs3 = vec![Segment::new(0, 0.0, 6.0, "long talk"), Segment::new(1, 6.0, 7.0, "jingle")];
        let mut bg3 = tone(8.0, 0.0);
        for v in bg3.samples.iter_mut().skip(sr as usize * 6) {
            *v = 0.3;
        }
        assert_eq!(classify_singing(&mut segs3, &voice, &bg3, ContentMode::Auto), 0);
        // Fundo alto o tempo todo, mas voz falada (tom sempre mudando): é jogo/filme, não música.
        let glide: Vec<f32> = {
            let mut phase = 0.0f32;
            (0..sr as usize * 8)
                .map(|i| {
                    let f = 120.0 + 80.0 * ((i as f32 / sr as f32) * 9.0).sin();
                    phase += 2.0 * std::f32::consts::PI * f / sr as f32;
                    phase.sin() * 0.3
                })
                .collect()
        };
        let spoken = AudioBuffer::new(sr, glide);
        let loud_bg = tone(8.0, 0.4);
        let mut segs4 = vec![Segment::new(0, 0.0, 4.0, "dialogue over score"), Segment::new(1, 4.0, 8.0, "more dialogue")];
        assert_eq!(classify_singing(&mut segs4, &spoken, &loud_bg, ContentMode::Auto), 0);
    }

    #[test]
    fn one_speaker_near_the_gender_cutoff_keeps_one_voice() {
        let mk =
            |n: usize| (0..n).map(|i| Segment::new(i, i as f64 * 2.0, i as f64 * 2.0 + 2.0, "line here")).collect::<Vec<_>>();
        // Um narrador oscilando em torno de 155 Hz: uma voz só.
        let mut segs = mk(8);
        let f0s: Vec<Option<f32>> = [122.0, 167.0, 137.0, 168.0, 167.0, 154.0, 171.0, 152.0].iter().map(|f| Some(*f)).collect();
        cluster_by_pitch(&mut segs, &f0s);
        assert!(segs.iter().all(|s| s.speaker == segs[0].speaker));
        // Homem e mulher de verdade: duas vozes.
        let mut segs = mk(8);
        let f0s: Vec<Option<f32>> = [115.0, 120.0, 210.0, 118.0, 220.0, 205.0, 122.0, 215.0].iter().map(|f| Some(*f)).collect();
        cluster_by_pitch(&mut segs, &f0s);
        assert_ne!(segs[0].speaker, segs[2].speaker);
        assert_eq!(segs[2].speaker, segs[4].speaker);
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
            Ok(vec![Segment::new(0, 0.5, 1.5, "Hello world."), Segment::new(1, 2.0, 3.0, "Second line.")])
        }
    }
    impl Translator for Fake {
        fn translate(&self, texts: &[&str], _: &Lang, _: &Lang) -> Result<Vec<String>> {
            Ok(texts.iter().map(|t| format!("PT: {t}")).collect())
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
        assert_eq!(out.segments[0].target_text.as_deref(), Some("PT: Hello world."));
        assert!(out.mixed_wav.exists() && out.srt.exists());
        let mixed = AudioBuffer::read_wav(&out.mixed_wav).unwrap();
        assert!(mixed.duration() >= 3.0);
        let events: Vec<_> = rx.try_iter().collect();
        assert!(events.iter().any(|e| e.stage == Stage::Done));
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Separador falso: a "voz" é o próprio áudio de entrada e o "fundo" é um
    /// tom alto só depois de 2 s — ou seja, fala a seco e depois música.
    struct FakeSep;
    impl Engine for FakeSep {
        fn info(&self) -> EngineInfo {
            EngineInfo { name: "fake-sep".into(), model: "-".into(), device: "cpu".into() }
        }
    }
    impl Separator for FakeSep {
        fn separate(&self, audio: &Path, out_dir: &Path) -> Result<(PathBuf, PathBuf)> {
            let src = AudioBuffer::read_wav(audio)?;
            let sr = src.sample_rate;
            let voice = out_dir.join("voice.wav");
            let bg = out_dir.join("bg.wav");
            src.write_wav(&voice)?;
            let mut b = AudioBuffer::silence(sr, src.duration());
            for (i, v) in b.samples.iter_mut().enumerate().skip(sr as usize * 2) {
                *v = ((i as f32) * 0.09).sin() * 0.5;
            }
            b.write_wav(&bg)?;
            Ok((voice, bg))
        }
    }
    struct FakeSung;
    impl Engine for FakeSung {
        fn info(&self) -> EngineInfo {
            Fake.info()
        }
    }
    impl Transcriber for FakeSung {
        fn transcribe(&self, _: &Path, _: Option<&Lang>) -> Result<Vec<Segment>> {
            // 0–2 s fala a seco; 2–8 s canto (fundo alto) num bloco longo sem pontuação.
            Ok(vec![
                Segment::new(0, 0.2, 1.8, "spoken intro words"),
                Segment::new(1, 2.0, 8.0, "sung words keep going and going without any punctuation at all here"),
            ])
        }
    }

    /// Separador falso com música alta o tempo todo.
    struct LoudSep;
    impl Engine for LoudSep {
        fn info(&self) -> EngineInfo {
            EngineInfo { name: "loud-sep".into(), model: "-".into(), device: "cpu".into() }
        }
    }
    impl Separator for LoudSep {
        fn separate(&self, audio: &Path, out_dir: &Path) -> Result<(PathBuf, PathBuf)> {
            let src = AudioBuffer::read_wav(audio)?;
            let sr = src.sample_rate;
            let voice = out_dir.join("voice.wav");
            let bg = out_dir.join("bg.wav");
            src.write_wav(&voice)?;
            let n = src.samples.len();
            AudioBuffer::new(sr, (0..n).map(|i| ((i as f32) * 0.011).sin() * 0.6).collect()).write_wav(&bg)?;
            Ok((voice, bg))
        }
    }

    #[test]
    fn dubbed_voice_sits_above_loud_music() {
        let dir = std::env::temp_dir().join(format!("dub-level-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("in.wav");
        let sr = 16000u32;
        AudioBuffer::new(sr, (0..sr as usize * 4).map(|i| ((i as f32) * 0.05).sin() * 0.3).collect()).write_wav(&src).unwrap();
        let f = Arc::new(Fake);
        let engines = EngineSet {
            transcriber: f.clone(),
            translator: f.clone(),
            synthesizer: f.clone(),
            separator: Some(Arc::new(LoudSep)),
            diarizer: None,
        };
        let cfg = PipelineConfig {
            sample_rate: sr,
            align_to_voice: false,
            content: ContentMode::Speech,
            match_loudness: false,
            ..Default::default()
        };
        let out = Pipeline::new(engines, cfg).run(&DubJob { audio: src, audio_hq: None, work_dir: dir.clone() }, None).unwrap();
        // Mede no próprio resultado: voz (trilha dublada escalada) vs fundo.
        let (_, chans) = AudioBuffer::read_wav_channels(&out.mixed_wav).unwrap();
        let mixed = AudioBuffer::new(sr, chans[0].clone());
        let win = [(out.segments[0].start + 0.1, out.segments[0].start + 0.6)];
        let total = mixed.rms_in_windows(&win);
        // Fundo nesse trecho: 0,6 de seno, abaixado pelo ducking (0,5) e pelo ganho (0,9).
        let bg = 0.6 * 0.707 * 0.5 * 0.9;
        let voice = (total * total - bg * bg).max(0.0).sqrt();
        let diff = db(voice) - db(bg);
        assert!(diff > 4.0, "voz deveria ficar ~6 dB acima do fundo, ficou {diff:.1} dB");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn music_mode_keeps_original_voice_and_skips_tts() {
        let dir = std::env::temp_dir().join(format!("dub-music-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("in.wav");
        let sr = 16000u32;
        // Voz original: tom constante de 0,3 ao longo de 8 s.
        AudioBuffer::new(sr, (0..sr as usize * 8).map(|i| ((i as f32) * 0.05).sin() * 0.3).collect()).write_wav(&src).unwrap();

        let run = |match_loudness: bool| {
            let f = Arc::new(Fake);
            let engines = EngineSet {
                transcriber: Arc::new(FakeSung),
                translator: f.clone(),
                synthesizer: f.clone(),
                separator: Some(Arc::new(FakeSep)),
                diarizer: None,
            };
            let cfg = PipelineConfig {
                sample_rate: sr,
                align_to_voice: false,
                max_segment_secs: 3.0,
                match_loudness,
                content: ContentMode::Auto,
                ..Default::default()
            };
            Pipeline::new(engines, cfg).run(&DubJob { audio: src.clone(), audio_hq: None, work_dir: dir.clone() }, None).unwrap()
        };
        let out = run(false);

        // O bloco de 6 s foi quebrado e tudo depois de 2 s virou canto.
        assert!(out.segments.len() >= 3, "{:?}", out.segments.len());
        assert!(!out.segments[0].is_singing());
        assert!(out.segments[0].dubbed_wav.is_some());
        assert!(out.segments[1..].iter().all(|s| s.is_singing() && s.dubbed_wav.is_none()));
        // Tradução existe para a legenda mesmo nos cantados.
        assert!(out.segments.iter().all(|s| s.target_text.is_some()));
        // Na mix, os trechos cantados têm a voz original somada ao fundo.
        let (_, chans) = AudioBuffer::read_wav_channels(&out.mixed_wav).unwrap();
        let mixed = AudioBuffer::new(sr, chans[0].clone());
        let sung = mixed.slice_seconds(4.0, 7.0).rms();
        let bg_only = 0.9 * AudioBuffer::new(sr, (0..sr as usize * 3).map(|i| ((i as f32) * 0.09).sin() * 0.5).collect()).rms();
        assert!(sung > bg_only * 1.1, "canto {sung} vs fundo {bg_only}");

        // Com o volume casado, a mix fica no nível do original (0,3/√2 ≈ 0,21).
        let out = run(true);
        let (_, chans) = AudioBuffer::read_wav_channels(&out.mixed_wav).unwrap();
        let whole = AudioBuffer::new(sr, chans[0].clone()).rms();
        assert!((whole - 0.212).abs() < 0.04, "rms {whole}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn srt_format() {
        let s = to_srt(&[Segment::new(0, 1.5, 2.25, "x")]);
        assert!(s.starts_with("1\n00:00:01,500 --> 00:00:02,250\nx\n"));
    }
}
