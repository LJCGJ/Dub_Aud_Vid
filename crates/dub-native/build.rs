use std::path::Path;

fn main() {
    let native = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../native");
    cxx_build::bridge("src/stretch.rs")
        .file(native.join("timestretch/timestretch.cpp"))
        .include(native.join("timestretch"))
        .std("c++17")
        .opt_level(3)
        .compile("dub_native_timestretch");

    println!("cargo:rerun-if-changed=src/stretch.rs");
    println!("cargo:rerun-if-changed=../../native/timestretch/timestretch.cpp");
    println!("cargo:rerun-if-changed=../../native/timestretch/timestretch.h");
}
