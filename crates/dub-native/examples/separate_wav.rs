//! `cargo run -p dub-native --example separate_wav -- <modelo> <mix.wav> <pasta_saida>`
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let sep = dub_native::MdxSeparator::load(&a[1], dub_core::Device::Cpu, 4).expect("load");
    let t = std::time::Instant::now();
    let (v, b) = dub_core::Separator::separate(&sep, std::path::Path::new(&a[2]), std::path::Path::new(&a[3])).expect("separate");
    println!("ok {:?} {:?} em {:.1}s", v, b, t.elapsed().as_secs_f32());
}
