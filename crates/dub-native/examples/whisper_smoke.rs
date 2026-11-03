//! `cargo run -p dub-native --example whisper_smoke -- <modelo.bin> <audio.wav>`
use dub_core::{Device, Lang, Transcriber};
use dub_native::WhisperTranscriber;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let t = WhisperTranscriber::load(&a[1], Device::Cpu, 4).expect("load");
    let segs = t
        .transcribe(std::path::Path::new(&a[2]), Some(&Lang::from(std::env::var("L").as_deref().unwrap_or("en"))))
        .expect("transcribe");
    for s in segs {
        println!("[{:6.2} → {:6.2}] {}", s.start, s.end, s.source_text);
    }
}
