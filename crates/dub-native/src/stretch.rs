//! Ponte `cxx` para o time-stretch WSOLA em C++ (`native/timestretch`).

use dub_core::pipeline::Stretcher;
use dub_core::{AudioBuffer, Result};

#[cxx::bridge(namespace = "dub::bridge")]
mod ffi {
    unsafe extern "C++" {
        include!("dub-native/src/stretch_shim.h");

        /// Estica `input` por `speed` (>1 encurta) preservando o tom.
        fn wsola(input: &[f32], sample_rate: u32, speed: f32) -> Vec<f32>;
    }
}

/// `Stretcher` nativo: WSOLA em C++, sem processos externos.
/// Qualidade boa na faixa 0.7x–1.5x, que é a usada pelo pipeline.
#[derive(Debug, Default, Clone, Copy)]
pub struct NativeStretcher;

impl NativeStretcher {
    pub fn stretch_samples(samples: &[f32], sample_rate: u32, speed: f32) -> Vec<f32> {
        ffi::wsola(samples, sample_rate, speed)
    }
}

impl Stretcher for NativeStretcher {
    fn stretch(&self, audio: &AudioBuffer, speed: f32) -> Result<AudioBuffer> {
        if (speed - 1.0).abs() < 1e-3 || audio.is_empty() {
            return Ok(audio.clone());
        }
        let out = ffi::wsola(&audio.samples, audio.sample_rate, speed);
        Ok(AudioBuffer::new(audio.sample_rate, out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(sr: u32, secs: f32, hz: f32) -> AudioBuffer {
        let n = (sr as f32 * secs) as usize;
        AudioBuffer::new(sr, (0..n).map(|i| (i as f32 * hz * std::f32::consts::TAU / sr as f32).sin() * 0.5).collect())
    }

    /// Estima a frequência dominante contando cruzamentos por zero.
    fn zero_cross_hz(a: &AudioBuffer) -> f32 {
        let crossings = a.samples.windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count();
        crossings as f32 / 2.0 / a.duration() as f32
    }

    #[test]
    fn duration_follows_speed() {
        let a = tone(16000, 2.0, 220.0);
        for speed in [0.8f32, 1.0, 1.25, 1.5] {
            let b = NativeStretcher.stretch(&a, speed).unwrap();
            let expected = a.duration() / speed as f64;
            assert!((b.duration() - expected).abs() < 0.01, "speed {speed}: {} vs {expected}", b.duration());
        }
    }

    #[test]
    fn pitch_is_preserved() {
        let a = tone(16000, 2.0, 220.0);
        let b = NativeStretcher.stretch(&a, 1.3).unwrap();
        let hz = zero_cross_hz(&b);
        // Reamostragem linear daria ~286 Hz; WSOLA deve ficar perto de 220.
        assert!((hz - 220.0).abs() < 8.0, "frequência após stretch: {hz}");
    }

    #[test]
    fn no_clipping_or_nan() {
        let a = tone(22050, 1.0, 440.0);
        let b = NativeStretcher.stretch(&a, 1.35).unwrap();
        assert!(b.samples.iter().all(|s| s.is_finite() && s.abs() <= 1.0));
    }

    #[test]
    fn short_input_does_not_panic() {
        let a = AudioBuffer::new(16000, vec![0.1; 100]);
        let b = NativeStretcher.stretch(&a, 1.2).unwrap();
        assert!(!b.is_empty());
    }
}
