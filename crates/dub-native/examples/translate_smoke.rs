//! `cargo run -p dub-native --example translate_smoke -- <modelo> "frase 1" "frase 2" ...`
use dub_core::{Device, Lang, Translator};
use dub_native::Ct2Translator;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let t0 = std::time::Instant::now();
    let t = Ct2Translator::load(&a[1], Device::Cpu, 2).expect("load");
    eprintln!("carregou em {:?}", t0.elapsed());
    let texts: Vec<&str> = a[2..].iter().map(|s| s.as_str()).collect();
    let t1 = std::time::Instant::now();
    let out = t.translate(&texts, &Lang::en(), &Lang::pt_br()).expect("translate");
    eprintln!("traduziu {} frases em {:?}", texts.len(), t1.elapsed());
    for (s, d) in texts.iter().zip(out) {
        println!("{s}\n  → {d}");
    }
}
