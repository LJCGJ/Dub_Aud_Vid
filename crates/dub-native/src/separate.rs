//! Separação voz/instrumental nativa com MDX-Net (modelos do Ultimate Vocal
//! Remover em ONNX), rodando no mesmo ONNX Runtime do Piper — CPU ou GPU
//! (DirectML/CUDA) sem Python.
//!
//! O modelo trabalha no domínio da frequência: recebe o espectrograma complexo
//! de um trecho estéreo a 44,1 kHz como tensor `[1, 4, dim_f, dim_t]`
//! (2 canais × re/im, `dim_f` primeiras raias, `dim_t` quadros) e devolve o
//! espectrograma do *stem* principal (voz, no Kim_Vocal_2). O instrumental é o
//! resto: `mix − voz`. O áudio inteiro é processado em janelas com sobreposição
//! e as bordas de cada janela são descartadas, como no UVR.
//!
//! Validado contra uma implementação de referência em NumPy: voz isolada com
//! correlação 0,99 com a fala original e instrumental 0,998 com a música.

use crate::piper::{effective_device, ort_err, OrtEngine};
use dub_core::engines::{Engine, EngineInfo, Separator};
use dub_core::paths::{self, ModelKind};
use dub_core::{AudioBuffer, Device, DubError, Result};
use ort::value::Tensor;
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info};

/// Hiperparâmetros de um modelo MDX-Net (do `model_data.json` do UVR).
#[derive(Debug, Clone, Copy)]
pub struct MdxParams {
    pub n_fft: usize,
    pub hop: usize,
    pub dim_f: usize,
    pub dim_t: usize,
    /// Fator de compensação de volume do stem principal.
    pub compensate: f32,
    /// `true` se o stem principal é a voz; `false` se é o instrumental.
    pub primary_is_vocals: bool,
}

impl MdxParams {
    /// Parâmetros por nome de modelo. Os valores vêm do UVR; modelos fora da
    /// lista usam o padrão mais comum (6144/1024/3072/256, stem = voz).
    pub fn for_model(name: &str) -> Self {
        let base = MdxParams { n_fft: 6144, hop: 1024, dim_f: 3072, dim_t: 256, compensate: 1.0, primary_is_vocals: true };
        let n = name.to_ascii_lowercase();
        if n.starts_with("kim_vocal_2") {
            MdxParams { compensate: 1.009, ..base }
        } else if n.starts_with("kim_vocal_1") {
            MdxParams { compensate: 1.035, ..base }
        } else if n.contains("inst_hq_3") || n.contains("inst_hq_4") {
            MdxParams { compensate: 1.022, primary_is_vocals: false, ..base }
        } else if n.contains("inst") {
            MdxParams { compensate: 1.035, primary_is_vocals: false, ..base }
        } else {
            base
        }
    }

    fn chunk(&self) -> usize {
        self.hop * (self.dim_t - 1)
    }
    fn trim(&self) -> usize {
        self.n_fft / 2
    }
}

pub struct MdxSeparator {
    engine: OrtEngine,
    params: MdxParams,
    name: String,
}

const MODEL_SR: u32 = 44_100;

impl MdxSeparator {
    /// `model` é um nome do catálogo (`Kim_Vocal_2`) ou um caminho `.onnx`.
    pub fn load(model: &str, device: Device, threads: usize) -> Result<Self> {
        let path = resolve_model(model)?;
        let device = effective_device(device);
        info!(model = %path.display(), %device, "carregando separador MDX-Net");
        let engine = OrtEngine::open("mdx-net", &path, device, threads)?;
        let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| model.to_string());
        let params = MdxParams::for_model(&name);
        Ok(Self { engine, params, name })
    }

    pub fn model_available(model: &str) -> bool {
        resolve_model(model).is_ok()
    }

    /// Separa um sinal estéreo a 44,1 kHz e devolve a voz como `[esquerda, direita]`
    /// (o instrumental é `mix − voz`).
    pub fn separate_stereo(&self, mix: &[Vec<f32>; 2], on_progress: &dyn Fn(usize, usize)) -> Result<[Vec<f32>; 2]> {
        let p = self.params;
        let len = mix[0].len().min(mix[1].len());
        let chunk = p.chunk();
        let trim = p.trim();
        let gen = chunk - 2 * trim;

        // Entrada com `trim` zeros à esquerda e sobra à direita, como no UVR.
        let padded_len = len + 2 * trim + gen;
        let mut padded = [vec![0.0f32; padded_len], vec![0.0f32; padded_len]];
        for c in 0..2 {
            padded[c][trim..trim + len].copy_from_slice(&mix[c][..len]);
        }
        let mut out = [vec![0.0f32; padded_len], vec![0.0f32; padded_len]];

        let fft = Stft::new(p.n_fft, p.hop);
        let total_chunks = (len + trim).div_ceil(gen);
        let mut i = 0usize;
        let mut done = 0usize;
        while i < len + trim {
            let seg: [&[f32]; 2] = [&padded[0][i..(i + chunk).min(padded_len)], &padded[1][i..(i + chunk).min(padded_len)]];
            let spec = fft.stft_pair(seg, chunk, p.dim_f);
            let out_spec: Vec<f32> = self.engine.run(|session| {
                let input = Tensor::from_array(([1usize, 4, p.dim_f, p.dim_t], spec.clone())).map_err(ort_err)?;
                let outputs = session.run(ort::inputs!["input" => input]).map_err(ort_err)?;
                let (_shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(ort_err)?;
                Ok(data.to_vec())
            })?;
            let y = fft.istft_pair(&out_spec, chunk, p.dim_f, p.dim_t);
            for c in 0..2 {
                let dst = &mut out[c][i + trim..i + chunk - trim];
                let src = &y[c][trim..chunk - trim];
                for (d, s) in dst.iter_mut().zip(src) {
                    *d += s;
                }
            }
            i += gen;
            done += 1;
            on_progress(done.min(total_chunks), total_chunks);
            debug!(chunk = done, total = total_chunks, "mdx");
        }

        let primary: [Vec<f32>; 2] = [
            out[0][trim..trim + len].iter().map(|v| v * p.compensate).collect(),
            out[1][trim..trim + len].iter().map(|v| v * p.compensate).collect(),
        ];
        if p.primary_is_vocals {
            Ok(primary)
        } else {
            // O modelo devolveu o instrumental: a voz é o resto.
            Ok([
                mix[0][..len].iter().zip(&primary[0]).map(|(m, s)| m - s).collect(),
                mix[1][..len].iter().zip(&primary[1]).map(|(m, s)| m - s).collect(),
            ])
        }
    }
}

fn resolve_model(model: &str) -> Result<PathBuf> {
    let p = Path::new(model);
    if p.extension().is_some() && p.is_file() {
        return Ok(p.to_path_buf());
    }
    paths::find(ModelKind::Separate, &format!("{model}.onnx")).ok_or_else(|| DubError::EngineUnavailable {
        engine: "mdx-net".into(),
        reason: format!(
            "modelo de separação '{model}' não encontrado em {:?} (rode: dub models download {model})",
            paths::model_dirs(ModelKind::Separate)
        ),
    })
}

/// STFT/iSTFT como o `torch.stft(center=True)` que o UVR usa: janela de Hann
/// periódica, preenchimento por reflexão de `n_fft/2` nas bordas e
/// normalização pela soma das janelas ao quadrado na reconstrução.
struct Stft {
    n_fft: usize,
    hop: usize,
    window: Vec<f32>,
    fwd: Arc<dyn RealToComplex<f32>>,
    inv: Arc<dyn ComplexToReal<f32>>,
}

impl Stft {
    fn new(n_fft: usize, hop: usize) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let window = (0..n_fft).map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n_fft as f32).cos()).collect();
        Self { n_fft, hop, window, fwd: planner.plan_fft_forward(n_fft), inv: planner.plan_fft_inverse(n_fft) }
    }

    fn frames(&self, len: usize) -> usize {
        1 + len / self.hop
    }

    /// Espectrograma `[4, dim_f, T]` (re/im do canal 0, re/im do canal 1) de um
    /// par de canais com `len` amostras (zeros se o trecho for mais curto).
    fn stft_pair(&self, chans: [&[f32]; 2], len: usize, dim_f: usize) -> Vec<f32> {
        let half = self.n_fft / 2;
        let t = self.frames(len);
        let mut out = vec![0.0f32; 4 * dim_f * t];
        let mut buf = vec![0.0f32; self.n_fft];
        let mut spec = self.fwd.make_output_vec();
        for (c, ch) in chans.iter().enumerate() {
            // Sinal com padding por reflexão.
            let mut x = vec![0.0f32; len + 2 * half];
            for (i, v) in x.iter_mut().enumerate() {
                let j = i as isize - half as isize;
                let idx = reflect(j, len as isize);
                *v = ch.get(idx).copied().unwrap_or(0.0);
            }
            for f in 0..t {
                for k in 0..self.n_fft {
                    buf[k] = x[f * self.hop + k] * self.window[k];
                }
                self.fwd.process(&mut buf, &mut spec).expect("fft");
                for (bin, s) in spec.iter().take(dim_f).enumerate() {
                    out[((2 * c) * dim_f + bin) * t + f] = s.re;
                    out[((2 * c + 1) * dim_f + bin) * t + f] = s.im;
                }
            }
        }
        out
    }

    /// Inverso de `stft_pair`: `[4, dim_f, T]` → dois canais de `len` amostras.
    fn istft_pair(&self, spec: &[f32], len: usize, dim_f: usize, t: usize) -> [Vec<f32>; 2] {
        let half = self.n_fft / 2;
        let n_bins = self.n_fft / 2 + 1;
        let total = self.n_fft + self.hop * (t - 1);
        let mut wsum = vec![0.0f32; total];
        for f in 0..t {
            for k in 0..self.n_fft {
                wsum[f * self.hop + k] += self.window[k] * self.window[k];
            }
        }
        let mut result = [Vec::new(), Vec::new()];
        let mut cbuf = vec![Complex::new(0.0f32, 0.0); n_bins];
        let mut frame = self.inv.make_output_vec();
        let scale = 1.0 / self.n_fft as f32; // realfft não normaliza a inversa
        for c in 0..2 {
            let mut acc = vec![0.0f32; total];
            for f in 0..t {
                for (bin, cv) in cbuf.iter_mut().enumerate() {
                    *cv = if bin < dim_f {
                        Complex::new(spec[((2 * c) * dim_f + bin) * t + f], spec[((2 * c + 1) * dim_f + bin) * t + f])
                    } else {
                        Complex::new(0.0, 0.0)
                    };
                }
                // Nyquist/DC precisam de parte imaginária zero para a inversa real.
                cbuf[0].im = 0.0;
                cbuf[n_bins - 1].im = 0.0;
                self.inv.process(&mut cbuf, &mut frame).expect("ifft");
                for k in 0..self.n_fft {
                    acc[f * self.hop + k] += frame[k] * scale * self.window[k];
                }
            }
            for (a, w) in acc.iter_mut().zip(&wsum) {
                *a /= w.max(1e-8);
            }
            result[c] = acc[half..(half + len).min(total)].to_vec();
            result[c].resize(len, 0.0);
        }
        result
    }
}

/// Índice refletido (como `np.pad(mode='reflect')`): -1 → 1, len → len-2.
fn reflect(j: isize, len: isize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = 2 * (len - 1);
    let mut k = j.rem_euclid(period);
    if k >= len {
        k = period - k;
    }
    k as usize
}

impl Engine for MdxSeparator {
    fn info(&self) -> EngineInfo {
        EngineInfo { name: "mdx-net".into(), model: self.name.clone(), device: self.engine.device().to_string() }
    }
}

impl Separator for MdxSeparator {
    fn separate(&self, audio: &Path, out_dir: &Path) -> Result<(PathBuf, PathBuf)> {
        self.run(audio, out_dir, None)
    }

    fn separate_with_progress(
        &self,
        audio: &Path,
        out_dir: &Path,
        on_progress: Arc<dyn Fn(u8) + Send + Sync>,
    ) -> Result<(PathBuf, PathBuf)> {
        self.run(audio, out_dir, Some(on_progress))
    }
}

impl MdxSeparator {
    fn run(
        &self,
        audio: &Path,
        out_dir: &Path,
        on_progress: Option<Arc<dyn Fn(u8) + Send + Sync>>,
    ) -> Result<(PathBuf, PathBuf)> {
        std::fs::create_dir_all(out_dir)?;
        let (sr, chans) = read_wav_stereo(audio)?;
        let mix = [resample(&chans[0], sr, MODEL_SR), resample(&chans[1], sr, MODEL_SR)];
        info!(seconds = mix[0].len() as f32 / MODEL_SR as f32, "separando voz e instrumental (MDX-Net)");
        let started = std::time::Instant::now();
        let vocals = self.separate_stereo(&mix, &|done, total| {
            debug!(done, total, "separação");
            if let Some(cb) = &on_progress {
                cb(((done * 100) / total.max(1)) as u8);
            }
        })?;
        let instrumental: [Vec<f32>; 2] = [
            mix[0].iter().zip(&vocals[0]).map(|(m, v)| m - v).collect(),
            mix[1].iter().zip(&vocals[1]).map(|(m, v)| m - v).collect(),
        ];
        info!(elapsed_s = started.elapsed().as_secs_f32(), "separação concluída");

        // Voz: mono 16 kHz para o whisper. Fundo: estéreo 44,1 kHz para a mixagem.
        let voice_mono: Vec<f32> = vocals[0].iter().zip(&vocals[1]).map(|(l, r)| 0.5 * (l + r)).collect();
        let voice_path = out_dir.join("voice_16k.wav");
        AudioBuffer::new(MODEL_SR, voice_mono).resampled(16_000).write_wav(&voice_path)?;
        let bg_path = out_dir.join("background_44k.wav");
        write_wav_stereo(&bg_path, MODEL_SR, &instrumental)?;
        // Também a voz em 44,1 kHz, para referência/depuração.
        let _ = write_wav_stereo(&out_dir.join("vocals_44k.wav"), MODEL_SR, &vocals);
        Ok((voice_path, bg_path))
    }
}

fn read_wav_stereo(path: &Path) -> Result<(u32, [Vec<f32>; 2])> {
    let mut reader = hound::WavReader::open(path).map_err(|e| DubError::Audio(format!("abrir WAV {}: {e}", path.display())))?;
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
    let mut l = Vec::with_capacity(frames);
    let mut r = Vec::with_capacity(frames);
    for fr in data.chunks(ch) {
        l.push(fr[0]);
        r.push(if ch > 1 { fr[1] } else { fr[0] });
    }
    Ok((spec.sample_rate, [l, r]))
}

fn write_wav_stereo(path: &Path, sr: u32, chans: &[Vec<f32>; 2]) -> Result<()> {
    let spec = hound::WavSpec { channels: 2, sample_rate: sr, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut w = hound::WavWriter::create(path, spec).map_err(|e| DubError::Audio(e.to_string()))?;
    for (l, r) in chans[0].iter().zip(&chans[1]) {
        for v in [l, r] {
            w.write_sample((v.clamp(-1.0, 1.0) * 32767.0) as i16).map_err(|e| DubError::Audio(e.to_string()))?;
        }
    }
    w.finalize().map_err(|e| DubError::Audio(e.to_string()))
}

fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let ratio = from as f64 / to as f64;
    let n = (x.len() as f64 / ratio).round() as usize;
    (0..n)
        .map(|i| {
            let pos = i as f64 * ratio;
            let i0 = pos.floor() as usize;
            let frac = (pos - i0 as f64) as f32;
            let a = x.get(i0).copied().unwrap_or(0.0);
            let b = x.get(i0 + 1).copied().unwrap_or(a);
            a + (b - a) * frac
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_matches_numpy() {
        assert_eq!(reflect(-1, 10), 1);
        assert_eq!(reflect(-3, 10), 3);
        assert_eq!(reflect(10, 10), 8);
        assert_eq!(reflect(0, 10), 0);
        assert_eq!(reflect(9, 10), 9);
    }

    #[test]
    fn stft_roundtrip() {
        let n = 20_000;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin() * 0.5 + (i as f32 * 0.31).cos() * 0.2).collect();
        let s = Stft::new(6144, 1024);
        let spec = s.stft_pair([&x, &x], n, 3072);
        let y = s.istft_pair(&spec, n, 3072, s.frames(n));
        // dim_f = 3072 descarta a raia de Nyquist: erro pequeno mas não zero.
        let err = x.iter().zip(&y[0]).skip(100).take(n - 200).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(err < 0.01, "erro de reconstrução {err}");
    }
}
