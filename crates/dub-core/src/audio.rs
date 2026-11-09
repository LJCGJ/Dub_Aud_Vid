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

    /// Lê um WAV com todos os canais separados (`[canal][amostra]`), sem
    /// somar para mono. Para o fundo estéreo da mixagem.
    pub fn read_wav_channels(path: &Path) -> Result<(u32, Vec<Vec<f32>>)> {
        let mut reader =
            hound::WavReader::open(path).map_err(|e| DubError::Audio(format!("abrir WAV {}: {e}", path.display())))?;
        let spec = reader.spec();
        let ch = spec.channels.max(1) as usize;
        let data: Vec<f32> = match spec.sample_format {
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
        let frames = data.len() / ch;
        let mut chans = vec![Vec::with_capacity(frames); ch];
        for fr in data.chunks(ch) {
            for (c, v) in fr.iter().enumerate() {
                chans[c].push(*v);
            }
        }
        Ok((spec.sample_rate, chans))
    }

    /// Grava vários canais (todos do mesmo tamanho) como WAV PCM 16 bits.
    pub fn write_wav_channels(path: &Path, sample_rate: u32, chans: &[Vec<f32>]) -> Result<()> {
        let n = chans.iter().map(|c| c.len()).min().unwrap_or(0);
        let spec = hound::WavSpec {
            channels: chans.len() as u16,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec).map_err(|e| DubError::Audio(e.to_string()))?;
        for i in 0..n {
            for c in chans {
                w.write_sample((c[i].clamp(-1.0, 1.0) * 32767.0) as i16).map_err(|e| DubError::Audio(e.to_string()))?;
            }
        }
        w.finalize().map_err(|e| DubError::Audio(e.to_string()))
    }

    /// Reamostra por interpolação linear (suficiente para fala; para música
    /// o pipeline trabalha na taxa nativa sempre que pode).
    pub fn resampled(&self, target_sr: u32) -> AudioBuffer {
        if self.sample_rate == target_sr || self.samples.is_empty() {
            return AudioBuffer { sample_rate: target_sr, samples: self.samples.clone() };
        }
        let ratio = self.sample_rate as f64 / target_sr as f64;
        let n_out = (self.samples.len() as f64 / ratio).round() as usize;
        let mut out = Vec::with_capacity(n_out);
        for i in 0..n_out {
            let pos = i as f64 * ratio;
            let i0 = pos.floor() as usize;
            let frac = (pos - i0 as f64) as f32;
            let x0 = self.samples.get(i0).copied().unwrap_or(0.0);
            let x1 = self.samples.get(i0 + 1).copied().unwrap_or(x0);
            out.push(x0 + (x1 - x0) * frac);
        }
        AudioBuffer::new(target_sr, out)
    }

    /// Frequência fundamental mediana (Hz) dos quadros com voz, pelo método
    /// YIN (diferença normalizada cumulativa, limiar 0,15 — evita o erro de
    /// oitava da autocorrelação simples). `None` se não há quadros vozeados o
    /// bastante. Serve para estimar se a voz é masculina ou feminina.
    pub fn median_f0(&self) -> Option<f32> {
        let sr = self.sample_rate as f32;
        let w = (sr * 0.04) as usize; // 40 ms
        let tau_min = (sr / 400.0) as usize;
        let tau_max = (sr / 60.0) as usize;
        if self.samples.len() < w + tau_max + 1 {
            return None;
        }
        let mut f0s: Vec<f32> = Vec::new();
        let mut d = vec![0.0f32; tau_max + 1];
        let mut i = 0;
        while i + w + tau_max < self.samples.len() {
            let x = &self.samples[i..i + w + tau_max];
            i += w / 2;
            let rms = (x[..w].iter().map(|v| v * v).sum::<f32>() / w as f32).sqrt();
            if rms < 0.02 {
                continue;
            }
            // d(τ) = Σ (x[k] − x[k+τ])²
            #[allow(clippy::needless_range_loop)]
            for tau in 1..=tau_max {
                let mut acc = 0.0f32;
                for k in 0..w {
                    let diff = x[k] - x[k + tau];
                    acc += diff * diff;
                }
                d[tau] = acc;
            }
            // d'(τ) = d(τ) · τ / Σ_{1..τ} d
            let mut cum = 0.0f32;
            let mut best: Option<usize> = None;
            let mut dp_prev = f32::MAX;
            for (tau, &dt) in d.iter().enumerate().skip(1) {
                cum += dt;
                let dp = if cum > 0.0 { dt * tau as f32 / cum } else { 1.0 };
                if tau >= tau_min {
                    if best.is_some() {
                        // segue o vale até o mínimo local
                        if dp < dp_prev {
                            best = Some(tau);
                        } else {
                            break;
                        }
                    } else if dp < 0.15 {
                        best = Some(tau);
                    }
                }
                dp_prev = dp;
            }
            if let Some(tau) = best {
                f0s.push(sr / tau as f32);
            }
        }
        if f0s.len() < 5 {
            return None;
        }
        f0s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some(f0s[f0s.len() / 2])
    }

    /// "Brilho": fração da energia acima de ~2 kHz (filtro de um polo), 0..1.
    /// Serve para aproximar o timbre da dublagem ao da voz original.
    pub fn brightness(&self) -> Option<f32> {
        if self.samples.len() < 1024 {
            return None;
        }
        let rc = 1.0 / (2.0 * std::f32::consts::PI * 2000.0);
        let dt = 1.0 / self.sample_rate as f32;
        let alpha = rc / (rc + dt); // passa-altas de um polo
        let mut hp = 0.0f32;
        let mut prev = self.samples[0];
        let mut e_hp = 0.0f64;
        let mut e_all = 0.0f64;
        for &x in &self.samples {
            hp = alpha * (hp + x - prev);
            prev = x;
            e_hp += (hp * hp) as f64;
            e_all += (x * x) as f64;
        }
        if e_all <= 1e-9 {
            return None;
        }
        Some((e_hp / e_all).clamp(0.0, 1.0) as f32)
    }

    /// Inclina o espectro: `tilt` > 0 realça agudos, < 0 abafa (±0,6 no máximo).
    /// Um polo em ~1,5 kHz; não é um equalizador de estúdio, é um ajuste suave.
    pub fn tilt(&mut self, tilt: f32) {
        let k = tilt.clamp(-0.6, 0.6);
        if k.abs() < 1e-3 || self.samples.is_empty() {
            return;
        }
        let rc = 1.0 / (2.0 * std::f32::consts::PI * 1500.0);
        let dt = 1.0 / self.sample_rate as f32;
        let a = dt / (rc + dt); // passa-baixas de um polo
        let mut lp = self.samples[0];
        for x in self.samples.iter_mut() {
            lp += a * (*x - lp);
            let hp = *x - lp;
            *x += k * hp;
        }
    }

    /// Muda o tom por `ratio` (2.0 = uma oitava acima) mantendo a duração:
    /// estica com `stretcher` e reamostra de volta. Para ajustes de até ±4
    /// semitons a qualidade do WSOLA é boa; além disso a voz fica artificial.
    pub fn pitch_shifted(&self, ratio: f32, stretcher: &dyn crate::pipeline::Stretcher) -> Result<AudioBuffer> {
        if (ratio - 1.0).abs() < 0.01 || self.samples.is_empty() {
            return Ok(self.clone());
        }
        // Mais longo por `ratio` (speed = 1/ratio), depois tocado `ratio` vezes
        // mais rápido reinterpretando a taxa de amostragem.
        let longer = stretcher.stretch(self, 1.0 / ratio)?;
        let fast = AudioBuffer::new((self.sample_rate as f32 * ratio).round() as u32, longer.samples);
        Ok(fast.resampled(self.sample_rate))
    }

    /// Recorta `[start, end)` em segundos.
    pub fn slice_seconds(&self, start: f64, end: f64) -> AudioBuffer {
        let sr = self.sample_rate as f64;
        let a = ((start.max(0.0)) * sr) as usize;
        let b = ((end.max(start)) * sr) as usize;
        let a = a.min(self.samples.len());
        let b = b.min(self.samples.len());
        AudioBuffer::new(self.sample_rate, self.samples[a..b].to_vec())
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

    /// Nível RMS (0..1) do buffer inteiro; 0 para áudio vazio.
    pub fn rms(&self) -> f32 {
        rms_of(&self.samples)
    }

    /// Soma `other` apenas dentro das janelas `(início, fim)` em segundos, com
    /// rampas de `fade` s nas bordas. Usado para devolver a voz original nos
    /// trechos cantados, sem estalos onde a dublagem entra e sai.
    pub fn mix_windows(&mut self, other: &AudioBuffer, windows: &[(f64, f64)], gain: f32, fade: f64) -> Result<()> {
        if other.sample_rate != self.sample_rate {
            return Err(DubError::Audio(format!(
                "sample rate incompatível na mixagem: {} vs {}",
                self.sample_rate, other.sample_rate
            )));
        }
        let sr = self.sample_rate as f64;
        let fade_n = (fade * sr).max(1.0);
        for &(start, end) in windows {
            if end <= start {
                continue;
            }
            let s0 = ((start - fade) * sr).floor().max(0.0) as usize;
            let s1 = (((end + fade) * sr).ceil() as usize).min(other.samples.len());
            if s1 > self.samples.len() {
                self.samples.resize(s1, 0.0);
            }
            let (a, b) = (start * sr, end * sr);
            for i in s0..s1 {
                let t = i as f64;
                let w = if t < a {
                    ((t - (a - fade_n)) / fade_n).clamp(0.0, 1.0)
                } else if t > b {
                    (((b + fade_n) - t) / fade_n).clamp(0.0, 1.0)
                } else {
                    1.0
                } as f32;
                self.samples[i] += other.samples[i] * gain * w;
            }
        }
        Ok(())
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

pub fn rms_of(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|v| v * v).sum::<f32>() / samples.len() as f32).sqrt()
}

/// Limitador suave para vários canais: abaixo de `knee` nada muda; acima, a
/// curva comprime até `ceiling` (tanh), e o ganho é o mesmo em todos os canais
/// (calculado pelo pico conjunto) para a imagem estéreo não se mover. Ao
/// contrário de escalar a faixa inteira pelo pico, um estouro isolado não
/// abaixa o vídeo todo.
pub fn soft_limit_channels(chans: &mut [Vec<f32>], knee: f32, ceiling: f32) {
    let n = chans.iter().map(|c| c.len()).max().unwrap_or(0);
    let range = (ceiling - knee).max(1e-3);
    for i in 0..n {
        let peak = chans.iter().map(|c| c.get(i).map(|v| v.abs()).unwrap_or(0.0)).fold(0.0f32, f32::max);
        if peak <= knee {
            continue;
        }
        let limited = knee + range * ((peak - knee) / range).tanh();
        let g = limited / peak;
        for c in chans.iter_mut() {
            if let Some(v) = c.get_mut(i) {
                *v *= g;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soft_limit_keeps_quiet_and_caps_loud() {
        let mut ch = vec![vec![0.5f32, 1.4, -2.0, 0.1]];
        soft_limit_channels(&mut ch, 0.8, 0.95);
        assert_eq!(ch[0][0], 0.5);
        assert_eq!(ch[0][3], 0.1);
        assert!(ch[0][1] > 0.8 && ch[0][1] < 0.95);
        assert!(ch[0][2] < -0.8 && ch[0][2] > -0.95);
    }

    #[test]
    fn mix_windows_only_inside() {
        let mut base = AudioBuffer::silence(100, 3.0);
        let other = AudioBuffer::new(100, vec![1.0; 300]);
        base.mix_windows(&other, &[(1.0, 2.0)], 1.0, 0.1).unwrap();
        assert_eq!(base.samples[50], 0.0);
        assert!((base.samples[150] - 1.0).abs() < 1e-6);
        assert_eq!(base.samples[250], 0.0);
        let edge = base.samples[95];
        assert!(edge > 0.0 && edge < 1.0, "{edge}");
    }

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
    fn f0_of_pure_tones() {
        let sr = 16_000u32;
        let tone = |f: f32| {
            AudioBuffer::new(sr, (0..sr).map(|i| (2.0 * std::f32::consts::PI * f * i as f32 / sr as f32).sin() * 0.5).collect())
        };
        let m = tone(120.0).median_f0().unwrap();
        let f = tone(220.0).median_f0().unwrap();
        assert!((m - 120.0).abs() < 8.0, "{m}");
        assert!((f - 220.0).abs() < 12.0, "{f}");
        assert_eq!(crate::Gender::from_f0(m), crate::Gender::Male);
        assert_eq!(crate::Gender::from_f0(f), crate::Gender::Female);
        assert!(AudioBuffer::silence(sr, 1.0).median_f0().is_none());
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
