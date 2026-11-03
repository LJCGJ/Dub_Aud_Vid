//! `cargo run -p dub-native --example piper_smoke -- <voz.onnx|nome> <texto> <saida.wav> [speed]`
use dub_core::{Device, Lang, Synthesizer};
use dub_native::PiperSynthesizer;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let speed: f32 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let t0 = std::time::Instant::now();
    let s = PiperSynthesizer::load(&a[1], Device::Cpu, 4).expect("load");
    eprintln!("carregou em {:?}", t0.elapsed());
    let t1 = std::time::Instant::now();
    let audio = s.synthesize(&a[2], &Lang::pt_br(), None, speed).expect("synth");
    eprintln!("sintetizou {:.2}s de áudio em {:?} ({} Hz)", audio.duration(), t1.elapsed(), audio.sample_rate);
    audio.write_wav(&a[3]).expect("wav");
}
