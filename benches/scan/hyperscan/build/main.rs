use std::{env, path::PathBuf};

mod source;

fn main() {
    // The benchmark can still run the Rust algorithms on other targets. A
    // workspace-wide check must not try to build x86 native code for them.
    if env::var("CARGO_CFG_TARGET_ARCH").unwrap() != "x86_64"
        || env::var("CARGO_CFG_TARGET_OS").unwrap() != "linux"
    {
        return;
    }
    for name in ["HYPERSCAN_SOURCE", "HYPERSCAN_BUILD", "BOOST_ROOT", "RAGEL"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    println!("cargo:rerun-if-changed=src/bridge.cpp");
    println!("cargo:rerun-if-changed=build");

    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let cache = root.join("../../../target/scan");
    let source = source::hyperscan(&cache);
    let boost = source::boost(&cache);
    let build = if let Some(path) = env::var_os("HYPERSCAN_BUILD") {
        PathBuf::from(path)
    } else {
        let ragel = source::ragel(&cache);
        // Use upstream's own targets and per-architecture compiler flags.
        let destination = cmake::Config::new(&source)
            // Let the upstream project provide the native compiler options.
            .no_default_flags(true)
            .profile("Release")
            .generator("Unix Makefiles")
            .define("CMAKE_POLICY_VERSION_MINIMUM", "3.5")
            .define("BOOST_ROOT", &boost)
            .define("RAGEL", ragel)
            .define("FAT_RUNTIME", "ON")
            .define("BUILD_AVX512", "ON")
            .define("BUILD_AVX512VBMI", "ON")
            .define("BUILD_EXAMPLES", "OFF")
            .build_target("hs_runtime")
            .build();
        destination.join("build")
    };
    assert!(build.join("lib/libhs_runtime.a").is_file(), "HYPERSCAN_BUILD must contain lib/libhs_runtime.a from a fat runtime build with AVX-512 and VBMI enabled");

    // Table construction is separate from timing. Compile upstream's two table
    // builders alongside the small ABI bridge, without building its regex compiler.
    let boost_include = if boost.join("include/boost").is_dir() {
        boost.join("include")
    } else {
        boost
    };
    cc::Build::new()
        .cpp(true)
        .std("c++11")
        // Upstream SIMD headers require SSSE3 even for the ABI bridge. Every
        // exposed backend is checked before entering this compiled code.
        .flag("-mssse3")
        .include(&build)
        .include(source.join("src"))
        .include(boost_include)
        .file(root.join("src/bridge.cpp"))
        .file(source.join("src/nfa/shufticompile.cpp"))
        .file(source.join("src/nfa/trufflecompile.cpp"))
        .define("NDEBUG", None)
        .warnings(false)
        .compile("scan_hyperscan_bridge");
    println!(
        "cargo:rustc-link-search=native={}",
        build.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static=hs_runtime");
}
