fn main() {
    let a: Vec<String> = std::env::args().collect();
    let w = dub_native::WhisperTranscriber::load_with_backend(&a[1], dub_core::Device::Cpu, 4, None).expect("load");
    let segs = dub_core::Transcriber::transcribe(&w, std::path::Path::new(&a[2]), Some(&dub_core::Lang::pt_br())).expect("t");
    for s in segs {
        println!("{:.1}-{:.1} {}", s.start, s.end, s.source_text);
    }
}
