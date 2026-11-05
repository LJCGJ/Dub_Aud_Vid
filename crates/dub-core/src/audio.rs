use crate::error::{DubError, Result};
use std::path::Path;

/// Áudio PCM mono em ponto flutuante (-1.0..1.0). Formato interno do pipeline.
#[derive(Debug, Clone, Default)]
pub struct AudioBuffer {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}

impl AudioBuffer {
    pub fn new(sample_rate: u32, samples: Vec<f32>) -> Self {
        Self { sample_rate, samples }
    }

    pub fn silence(sample_rate: u32, seconds: f64) -> Self {
        let n = (sample_rate as f64 * seconds).round().max(0.0) as usize;
        Self { sample_rate, samples: vec![0.0; n] }
    }

    pub fn duration(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.samples.len() as f64 / self.sample_rate as f64
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Lê um WAV (qualquer canal/bit depth) e converte para mono f32.
    pub fn read_wav(path: impl AsRef<Path>) -> Result<Self> {
        let mut reader = hound::WavReader::open(path.as_ref())
            .map_err(|e| DubError::Audio(format!("abrir WAV {}: {e}", path.as_ref().display())))?;
        let spec = reader.spec();
        let channels = spec.channels.max(1) as usize;

        let interleaved: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => {
                reader.samples::<f32>().collect::<std::result::Result<_, _>>().map_err(|e| DubError::Audio(e.to_string()))?
            }
            hound::SampleFormat::Int => {
                let max = ((1u64 << (spec.bits_per_sample - 1)) - 1) as f32;
                reader
                    .samples::<i32>()
                    .map(|s| s.map(|v| v as f32 / max))
                    .collect::<std::result::Result<_, _>>()
                    .map_err(|e| DubError::Audio(e.to_string()))?
            }
        };

        let samples = if channels == 1 {
            interleaved
        } else {
            interleaved.chunks(channels).map(|frame| frame.iter().sum::<f32>() / channels as f32).collect()
        };

        Ok(Self { sample_rate: spec.sample_rate, samples })
    }

    /// Grava como WAV mono 16-bit PCM.
    pub fn write_wav(&self, path: impl AsRef<Path>) -> Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path.as_ref(), spec)
            .map_err(|e| DubError::Audio(format!("criar WAV {}: {e}", path.as_ref().display())))?;
        for &s in &self.samples {
            let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            writer.write_sample(v).map_err(|e| DubError::Audio(e.to_string()))?;
        }
        writer.finalize().map_err(|e| DubError::Audio(e.to_string()))?;
        Ok(())
    }

    /// Soma `other` neste buffer começando em `at_seconds`, estendendo se preciso.
    /// Ambos precisam ter o mesmo sample rate.
    pub fn mix_at(&mut self, other: &AudioBuffer, at_seconds: f64, gain: f32) -> Result<()> {
        if other.sample_rate != self.sample_rate {
            return Err(DubError::Audio(format!(
                "sample rate incompatível na mixagem: {} vs {}",
                self.sample_rate, other.sample_rate
            )));
        }
        let offset = (at_seconds.max(0.0) * self.sample_rate as f64).round() as usize;
        let needed = offset + other.samples.len();
        if self.samples.len() < needed {
            self.samples.resize(needed, 0.0);
        }
        for (i, &s) in other.samples.iter().enumerate() {
            self.samples[offset + i] += s * gain;
        }
        Ok(())
    }

    /// *Ducking*: abaixa o volume para `gain` dentro de cada janela
    /// `(início, fim)` em segundos, com rampas de `fade` segundos nas bordas
    /// para não estalar. Fora das janelas o áudio fica intacto. Usado para
    /// manter música/ruído do original enquanto a voz dublada toca por cima.
    pub fn duck(&mut self, windows: &[(f64, f64)], gain: f32, fade: f64) {
        let sr = self.sample_rate as f64;
        let n = self.samples.len();
        if n == 0 || windows.is_empty() {
            return;
        }
        // Envelope por amostra (1.0 = sem alteração), calculado como o mínimo
        // entre as janelas, para sobreposições não "somarem" atenuação.
        let mut env = vec![1.0f32; n];
        let fade_n = (fade * sr).max(1.0);
        for &(start, end) in windows {
            if end <= start {
                continue;
            }
            let s0 = ((start - fade) * sr).floor().max(0.0) as usize;
            let s1 = ((end + fade) * sr).ceil().min(n as f64) as usize;
            let a = start * sr;
            let b = end * sr;
            for (i, e) in env.iter_mut().enumerate().take(s1).skip(s0) {
                let t = i as f64;
                // 0 fora, 1 dentro, rampa linear nas bordas.
                let w = if t < a {
                    ((t - (a - fade_n)) / fade_n).clamp(0.0, 1.0)
                } else if t > b {
                    (((b + fade_n) - t) / fade_n).clamp(0.0, 1.0)
                } else {
                    1.0
                } as f32;
                let g = 1.0 - w * (1.0 - gain);
                if g < *e {
                    *e = g;
                }
            }
        }
        for (s, e) in self.samples.iter_mut().zip(env) {
            *s *= e;
        }
    }

    /// Normaliza o pico para `peak` (ex.: 0.95) se estiver acima.
    pub fn limit_peak(&mut self, peak: f32) {
        let max = self.samples.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        if max > peak && max > 0.0 {
            let g = peak / max;
            self.samples.iter_mut().for_each(|s| *s *= g);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mix_extends_and_sums() {
        let mut base = AudioBuffer::silence(10, 1.0);
        let tone = AudioBuffer::new(10, vec![0.5; 5]);
        base.mix_at(&tone, 0.8, 1.0).unwrap();
        assert_eq!(base.samples.len(), 13);
        assert_eq!(base.samples[8], 0.5);
        assert_eq!(base.samples[7], 0.0);
    }

    #[test]
    fn duck_lowers_only_inside_windows() {
        let mut a = AudioBuffer::new(100, vec![1.0; 300]); // 3 s
        a.duck(&[(1.0, 2.0)], 0.25, 0.1);
        assert_eq!(a.samples[20], 1.0); // antes da rampa
        assert!((a.samples[150] - 0.25).abs() < 1e-6); // dentro
        assert_eq!(a.samples[280], 1.0); // depois
        let edge = a.samples[95]; // meio da rampa de entrada
        assert!(edge > 0.25 && edge < 1.0, "rampa: {edge}");
    }

    #[test]
    fn wav_roundtrip() {
        let dir = std::env::temp_dir().join(format!("dub-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.wav");
        let a = AudioBuffer::new(8000, (0..800).map(|i| (i as f32 / 800.0).sin() * 0.5).collect());
        a.write_wav(&p).unwrap();
        let b = AudioBuffer::read_wav(&p).unwrap();
        assert_eq!(b.sample_rate, 8000);
        assert_eq!(b.samples.len(), 800);
        assert!((a.samples[400] - b.samples[400]).abs() < 1e-3);
        let _ = std::fs::remove_dir_all(dir);
    }
}
