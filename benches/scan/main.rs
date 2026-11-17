use std::{env, path::Path, process::Command};

fn main() {
    // Keep native build dependencies in the benchmark workspace. Normal library
    // builds and unit tests do not download or compile Hyperscan.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let status = Command::new(env!("CARGO"))
        .args(["run", "--release", "--manifest-path"])
        .arg(root.join("benches/scan/Cargo.toml"))
        .args(["--target-dir"])
        .arg(root.join("target/scan"))
        .args(["--"])
        // Cargo adds this flag to harness-free benchmarks.
        .args(env::args_os().skip(1).filter(|arg| arg != "--bench"))
        .status()
        .expect("failed to start the scan benchmark");
    std::process::exit(status.code().unwrap_or(1));
}
