//! Traits que toda engine implementa. O pipeline só conhece estes traits;
//! quem é whisper.cpp, faster-whisper, Piper ou XTTS é detalhe do crate que
//! implementa.
//!
//! Todas as engines são `Send + Sync` porque rodam em threads de worker.

use crate::audio::AudioBuffer;
use crate::error::Result;
use crate::segment::{Lang, Segment, Speaker};
use std::path::Path;
use std::sync::Arc;

/// Informação sobre a engine para logs e UI.
#[derive(Debug, Clone)]
pub struct EngineInfo {
    pub name: String,
    pub model: String,
    pub device: String,
}

pub trait Engine: Send + Sync {
    fn info(&self) -> EngineInfo;
}

/// Converte áudio em segmentos com texto e timestamps.
pub trait Transcriber: Engine {
    /// `audio` é o WAV mono 16 kHz com a **voz isolada** (ou o original, se não houver separação).
    /// Deve devolver segmentos em ordem, com `index` sequencial a partir de 0.
    fn transcribe(&self, audio: &Path, lang_hint: Option<&Lang>) -> Result<Vec<Segment>>;

    /// Como `transcribe`, chamando `on_progress(0..=100)` conforme avança.
    /// Engines sem noção de progresso usam a implementação padrão (sem eventos).
    fn transcribe_with_progress(
        &self,
        audio: &Path,
        lang_hint: Option<&Lang>,
        _on_progress: Arc<dyn Fn(u8) + Send + Sync>,
    ) -> Result<Vec<Segment>> {
        self.transcribe(audio, lang_hint)
    }
}

/// Traduz texto entre idiomas.
pub trait Translator: Engine {
    /// Traduz em lote — a maioria das engines é muito mais rápida assim.
    /// Deve devolver exatamente `texts.len()` strings, na mesma ordem.
    fn translate(&self, texts: &[&str], from: &Lang, to: &Lang) -> Result<Vec<String>>;
}

/// Gera fala a partir de texto.
pub trait Synthesizer: Engine {
    /// Sintetiza `text` e devolve o áudio. `speaker_ref` é um WAV curto do
    /// falante original para engines com clonagem; engines sem clonagem ignoram.
    /// `speed` é a velocidade relativa (1.0 = normal) — engines que não suportam
    /// devolvem 1.0 e deixam o time-fit para o pipeline.
    fn synthesize(&self, text: &str, lang: &Lang, speaker_ref: Option<&Path>, speed: f32) -> Result<AudioBuffer>;

    /// Como `synthesize`, com o segmento à mão (gênero estimado, falante) para
    /// engines que escolhem a voz por trecho. O padrão ignora o segmento.
    fn synthesize_segment(
        &self,
        _segment: &Segment,
        text: &str,
        lang: &Lang,
        speaker_ref: Option<&Path>,
        speed: f32,
    ) -> Result<AudioBuffer> {
        self.synthesize(text, lang, speaker_ref, speed)
    }

    /// Se `true`, o pipeline passa `speaker_ref`; senão não perde tempo extraindo.
    fn supports_voice_cloning(&self) -> bool {
        false
    }

    /// Se `true`, o pipeline pede ao TTS para já falar na velocidade certa em vez de
    /// esticar depois (qualidade melhor).
    fn supports_speed(&self) -> bool {
        false
    }
}

/// Separa a voz do fundo (música/efeitos).
pub trait Separator: Engine {
    /// Devolve `(voz, fundo)` como caminhos de WAV gravados em `out_dir`.
    fn separate(&self, audio: &Path, out_dir: &Path) -> Result<(std::path::PathBuf, std::path::PathBuf)>;

    /// Como `separate`, chamando `on_progress(0..=100)` conforme avança.
    fn separate_with_progress(
        &self,
        audio: &Path,
        out_dir: &Path,
        _on_progress: Arc<dyn Fn(u8) + Send + Sync>,
    ) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
        self.separate(audio, out_dir)
    }
}

/// Identifica quem fala quando.
pub trait Diarizer: Engine {
    /// Para cada segmento, define `speaker` e `overlapped`. Pode mesclar/dividir
    /// segmentos se a engine devolver fronteiras melhores; mantém a ordem.
    fn diarize(&self, audio: &Path, segments: Vec<Segment>) -> Result<Vec<Segment>>;

    /// Lista de falantes encontrados (para a UI escolher uma voz por falante).
    fn speakers(&self) -> Vec<Speaker> {
        Vec::new()
    }
}

/// Conjunto de engines injetado no pipeline. `separator` e `diarizer` são
/// opcionais — sem eles o pipeline usa o áudio original e um único falante.
#[derive(Clone)]
pub struct EngineSet {
    pub transcriber: Arc<dyn Transcriber>,
    pub translator: Arc<dyn Translator>,
    pub synthesizer: Arc<dyn Synthesizer>,
    pub separator: Option<Arc<dyn Separator>>,
    pub diarizer: Option<Arc<dyn Diarizer>>,
}

impl EngineSet {
    pub fn describe(&self) -> Vec<(&'static str, EngineInfo)> {
        let mut v = vec![
            ("transcriber", self.transcriber.info()),
            ("translator", self.translator.info()),
            ("synthesizer", self.synthesizer.info()),
        ];
        if let Some(s) = &self.separator {
            v.push(("separator", s.info()));
        }
        if let Some(d) = &self.diarizer {
            v.push(("diarizer", d.info()));
        }
        v
    }
}
