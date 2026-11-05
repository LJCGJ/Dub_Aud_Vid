//! `cargo run -p dub-native --example kokoro_say -- <voz> "<texto>" saida.wav`
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let k = dub_native::KokoroSynthesizer::load(&a[1], dub_core::Device::Cpu, 4).expect("load");
    let t = std::time::Instant::now();
    let audio = dub_core::Synthesizer::synthesize(&k, &a[2], &dub_core::Lang::pt_br(), None, 1.0).expect("synth");
    audio.write_wav(&a[3]).expect("wav");
    println!("{:.2}s de áudio em {:.2}s", audio.duration(), t.elapsed().as_secs_f32());
}
