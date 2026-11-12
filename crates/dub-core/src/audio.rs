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

    /// Corta o silêncio do começo e do fim (abaixo de `threshold` de pico em
    /// janelas de 10 ms), deixando `keep` segundos de folga em cada ponta.
    pub fn trim_silence(&mut self, threshold: f32, keep: f64) {
        let win = (self.sample_rate as usize / 100).max(1);
        let n = self.samples.len();
        if n < win * 4 {
            return;
        }
        let loud = |i: usize| self.samples[i..(i + win).min(n)].iter().any(|v| v.abs() > threshold);
        let Some(first) = (0..n).step_by(win).find(|&i| loud(i)) else { return };
        let last = (0..n).step_by(win).rev().find(|&i| loud(i)).unwrap_or(first);
        let pad = (keep * self.sample_rate as f64) as usize;
        let a = first.saturating_sub(pad);
        let b = (last + win + pad).min(n);
        if a > 0 || b < n {
            self.samples = self.samples[a..b].to_vec();
        }
    }

    /// Trilha de tom: f0 (Hz) a cada 16 ms, 0.0 onde não há voz (YIN, janela
    /// de 32 ms, 70–500 Hz).
    pub fn f0_track(&self) -> Vec<f32> {
        let sr = self.sample_rate as f32;
        let w = (sr * 0.032) as usize;
        let hop = ((sr * 0.016) as usize).max(1);
        let tau_min = ((sr / 500.0) as usize).max(1);
        let tau_max = (sr / 70.0) as usize;
        let mut out = Vec::new();
        if w == 0 || self.samples.len() < w + tau_max + 1 {
            return out;
        }
        let mut i = 0;
        while i + w + tau_max < self.samples.len() {
            let x = &self.samples[i..i + w + tau_max];
            i += hop;
            let rms = (x[..w].iter().map(|v| v * v).sum::<f32>() / w as f32).sqrt();
            if rms < 0.015 {
                out.push(0.0);
                continue;
            }
            let mut cum = 0.0f32;
            let mut found = 0.0f32;
            for tau in 1..=tau_max {
                let mut acc = 0.0f32;
                for k in 0..w {
                    let d = x[k] - x[k + tau];
                    acc += d * d;
                }
                cum += acc;
                let dp = if cum > 0.0 { acc * tau as f32 / cum } else { 1.0 };
                if tau >= tau_min && dp < 0.15 {
                    found = sr / tau as f32;
                    break;
                }
            }
            out.push(found);
        }
        out
    }

    /// Fração dos quadros com voz que pertencem a "notas sustentadas": trechos
    /// de pelo menos ~190 ms em que o tom fica dentro de 1 semitom. Canto tem
    /// muitas; fala (narração, diálogo de jogo) quase nenhuma. Medido em voz
    /// isolada: canto ≈ 0,2 na mediana por trecho, fala ≈ 0,0.
    pub fn sustained_ratio(&self) -> Option<f32> {
        let f = self.f0_track();
        let voiced = f.iter().filter(|v| **v > 0.0).count();
        if voiced < 8 {
            return None;
        }
        let st: Vec<f32> = f.iter().map(|v| if *v > 0.0 { 12.0 * (v / 100.0).log2() } else { f32::NAN }).collect();
        let n = st.len();
        let mut steady = 0usize;
        let mut i = 0;
        while i < n {
            if st[i].is_nan() {
                i += 1;
                continue;
            }
            let (mut lo, mut hi) = (st[i], st[i]);
            let mut j = i;
            while j + 1 < n && !st[j + 1].is_nan() && (st[j + 1] - st[j]).abs() < 0.35 {
                let (nlo, nhi) = (lo.min(st[j + 1]), hi.max(st[j + 1]));
                if nhi - nlo >= 1.0 {
                    break;
                }
                lo = nlo;
                hi = nhi;
                j += 1;
            }
            if j - i + 1 >= 12 {
                steady += j - i + 1;
            }
            i = j + 1;
        }
        Some(steady as f32 / voiced as f32)
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

    /// Filtro passa-altas (biquad Butterworth de 2ª ordem): tira o grave
    /// "embolado" da voz sintetizada, que briga com o baixo da música.
    pub fn highpass(&mut self, cutoff_hz: f32) {
        let sr = self.sample_rate as f32;
        if cutoff_hz <= 0.0 || cutoff_hz >= sr * 0.45 {
            return;
        }
        let w0 = 2.0 * std::f32::consts::PI * cutoff_hz / sr;
        let (sin, cos) = w0.sin_cos();
        // Butterworth: Q = 1/√2.
        let alpha = sin / (2.0 * std::f32::consts::FRAC_1_SQRT_2);
        let b0 = (1.0 + cos) / 2.0;
        let b1 = -(1.0 + cos);
        let b2 = (1.0 + cos) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos;
        let a2 = 1.0 - alpha;
        biquad(&mut self.samples, [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0]);
    }

    /// Equalizador paramétrico (pico) em `freq_hz`, largura `q`, ganho em dB.
    /// +3 dB perto de 3 kHz dá "presença": a voz recorta a música sem precisar
    /// de muito mais volume.
    pub fn peaking_eq(&mut self, freq_hz: f32, q: f32, gain_db: f32) {
        let sr = self.sample_rate as f32;
        if gain_db.abs() < 0.01 || freq_hz <= 0.0 || freq_hz >= sr * 0.45 || q <= 0.0 {
            return;
        }
        let a = 10f32.powf(gain_db / 40.0);
        let w0 = 2.0 * std::f32::consts::PI * freq_hz / sr;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q);
        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * cos;
        let b2 = 1.0 - alpha * a;
        let a0 = 1.0 + alpha / a;
        let a1 = -2.0 * cos;
        let a2 = 1.0 - alpha / a;
        biquad(&mut self.samples, [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0]);
    }

    /// Compressor (detector de pico com ataque/relaxamento): acima de
    /// `threshold_db`, o excesso é dividido por `ratio`. Deixa o volume da fala
    /// estável, para ela não sumir nas sílabas fracas nem estourar nas fortes.
    /// Ganho de compensação não incluído (o nível final é ajustado depois).
    pub fn compress(&mut self, threshold_db: f32, ratio: f32, attack_ms: f32, release_ms: f32) {
        let sr = self.sample_rate as f32;
        if ratio <= 1.0 || self.samples.is_empty() {
            return;
        }
        let att = (-1.0 / (attack_ms.max(0.1) * 0.001 * sr)).exp();
        let rel = (-1.0 / (release_ms.max(1.0) * 0.001 * sr)).exp();
        let mut env_db = -120.0f32;
        for s in self.samples.iter_mut() {
            let lvl = 20.0 * s.abs().max(1e-6).log10();
            let coef = if lvl > env_db { att } else { rel };
            env_db = coef * env_db + (1.0 - coef) * lvl;
            let over = env_db - threshold_db;
            if over > 0.0 {
                let reduction_db = over - over / ratio;
                *s *= 10f32.powf(-reduction_db / 20.0);
            }
        }
    }

    /// Multiplica todas as amostras por `gain`.
    pub fn scale(&mut self, gain: f32) {
        self.samples.iter_mut().for_each(|s| *s *= gain);
    }

    /// RMS só dentro das janelas `(início, fim)` em segundos (0 se vazias).
    pub fn rms_in_windows(&self, windows: &[(f64, f64)]) -> f32 {
        rms_in_windows(&self.samples, self.sample_rate, windows)
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

/// Biquad em forma direta I. `c = [b0, b1, b2, a1, a2]` já normalizados por a0.
fn biquad(samples: &mut [f32], c: [f32; 5]) {
    let (mut x1, mut x2, mut y1, mut y2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for s in samples.iter_mut() {
        let x0 = *s;
        let y0 = c[0] * x0 + c[1] * x1 + c[2] * x2 - c[3] * y1 - c[4] * y2;
        x2 = x1;
        x1 = x0;
        y2 = y1;
        y1 = y0;
        *s = y0;
    }
}

/// RMS de `samples` só dentro das janelas `(início, fim)` em segundos.
pub fn rms_in_windows(samples: &[f32], sample_rate: u32, windows: &[(f64, f64)]) -> f32 {
    let sr = sample_rate as f64;
    let (mut sum, mut n) = (0.0f64, 0usize);
    for &(a, b) in windows {
        let i0 = ((a.max(0.0) * sr) as usize).min(samples.len());
        let i1 = ((b.max(a) * sr) as usize).min(samples.len());
        for v in &samples[i0..i1] {
            sum += (*v as f64) * (*v as f64);
        }
        n += i1 - i0;
    }
    if n == 0 {
        0.0
    } else {
        (sum / n as f64).sqrt() as f32
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

    fn sine(sr: u32, hz: f32, secs: f32, amp: f32) -> AudioBuffer {
        let n = (sr as f32 * secs) as usize;
        AudioBuffer::new(sr, (0..n).map(|i| amp * (2.0 * std::f32::consts::PI * hz * i as f32 / sr as f32).sin()).collect())
    }

    #[test]
    fn highpass_cuts_low_keeps_high() {
        let mut low = sine(44_100, 40.0, 1.0, 0.5);
        low.highpass(110.0);
        let mut high = sine(44_100, 2000.0, 1.0, 0.5);
        high.highpass(110.0);
        let tail = |a: &AudioBuffer| rms_of(&a.samples[a.samples.len() / 2..]);
        assert!(tail(&low) < 0.5 * 0.707 * 0.2, "40 Hz deveria cair bastante: {}", tail(&low));
        assert!((tail(&high) - 0.5 * 0.707).abs() < 0.02, "2 kHz deveria passar: {}", tail(&high));
    }

    #[test]
    fn peaking_eq_boosts_center() {
        let mut a = sine(44_100, 3000.0, 1.0, 0.1);
        a.peaking_eq(3000.0, 1.0, 6.0);
        let r = rms_of(&a.samples[22_050..]) / (0.1 * 0.707);
        assert!((r - 2.0).abs() < 0.1, "+6 dB ≈ x2: {r}");
    }

    #[test]
    fn compressor_reduces_loud_keeps_quiet() {
        let mut loud = sine(44_100, 500.0, 1.0, 0.9);
        loud.compress(-24.0, 3.0, 5.0, 120.0);
        let mut quiet = sine(44_100, 500.0, 1.0, 0.02);
        quiet.compress(-24.0, 3.0, 5.0, 120.0);
        assert!(rms_of(&loud.samples[22_050..]) < 0.9 * 0.707 * 0.4);
        assert!((rms_of(&quiet.samples[22_050..]) - 0.02 * 0.707).abs() < 0.002);
    }

    #[test]
    fn rms_in_windows_measures_only_windows() {
        let mut a = AudioBuffer::silence(100, 4.0);
        for v in &mut a.samples[100..200] {
            *v = 0.5;
        }
        assert!((a.rms_in_windows(&[(1.0, 2.0)]) - 0.5).abs() < 1e-6);
        assert_eq!(a.rms_in_windows(&[(2.5, 3.5)]), 0.0);
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
    fn trim_silence_removes_edges() {
        let sr = 16_000u32;
        let mut x = vec![0.0f32; sr as usize];
        for v in x.iter_mut().take(12_000).skip(4_000) {
            *v = 0.5;
        }
        let mut a = AudioBuffer::new(sr, x);
        a.trim_silence(0.01, 0.02);
        assert!((a.duration() - 0.54).abs() < 0.03, "{}", a.duration());
        assert!(a.samples[400].abs() > 0.4);
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
