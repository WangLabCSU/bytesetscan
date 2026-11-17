use std::{
    env,
    fs::{self, File},
    io,
    path::{Path, PathBuf},
    process::Command,
};

pub(super) fn hyperscan(cache: &Path) -> PathBuf {
    if let Some(path) = env::var_os("HYPERSCAN_SOURCE") {
        return PathBuf::from(path);
    }
    let revision = "bc3b191ab56055e8560c7cdc161c289c4d76e3d2";
    let destination = cache.join("sources").join(revision);
    if !destination.join("CMakeLists.txt").is_file() {
        let archive = download(
            cache,
            &format!("hyperscan-{revision}.tar.gz"),
            &format!("https://codeload.github.com/intel/hyperscan/tar.gz/{revision}"),
        );
        unpack(&archive, &destination, &[]);
    }
    destination
        .canonicalize()
        .expect("Hyperscan source directory")
}

pub(super) fn boost(cache: &Path) -> PathBuf {
    if let Some(path) = env::var_os("BOOST_ROOT") {
        return PathBuf::from(path);
    }
    if Path::new("/usr/include/boost/version.hpp").is_file() {
        return PathBuf::from("/usr");
    }
    let destination = cache.join("dependencies/boost-1.90.0");
    if !destination.join("include/boost/version.hpp").is_file()
        && !destination.join("boost/version.hpp").is_file()
    {
        let archive = download(
            cache,
            "boost_1_90_0.tar.bz2",
            "https://archives.boost.io/release/1.90.0/source/boost_1_90_0.tar.bz2",
        );
        // Only headers and their license are needed; avoid extracting Boost's
        // tests, examples, documentation, and library source trees.
        unpack(
            &archive,
            &destination,
            &["boost_1_90_0/boost", "boost_1_90_0/LICENSE_1_0.txt"],
        );
    }
    destination.canonicalize().expect("Boost headers directory")
}

pub(super) fn ragel(cache: &Path) -> PathBuf {
    if let Some(path) = env::var_os("RAGEL") {
        return PathBuf::from(path);
    }
    if let Some(paths) = env::var_os("PATH") {
        for directory in env::split_paths(&paths) {
            let binary = directory.join("ragel");
            if binary.is_file()
                && Command::new(&binary)
                    .arg("-v")
                    .output()
                    .is_ok_and(|out| out.status.success())
            {
                return binary.canonicalize().expect("system Ragel binary");
            }
        }
    }
    let destination = cache.join("dependencies/ragel-6.11");
    let binary = destination.join("ragel/ragel");
    if !binary.is_file() {
        let archive = download(
            cache,
            "ragel-6.11.tar.gz",
            "https://www.colm.net/files/ragel/ragel-6.11.tar.gz",
        );
        unpack(&archive, &destination, &[]);
        run(Command::new("./configure")
            .arg("--disable-manual")
            .current_dir(&destination));
        run(Command::new("make")
            .args(["-C", "ragel", "-j2", "ragel"])
            .current_dir(&destination));
    }
    binary.canonicalize().expect("Ragel binary")
}

fn download(cache: &Path, name: &str, url: &str) -> PathBuf {
    let archives = cache.join("dependencies/archives");
    fs::create_dir_all(&archives).expect("create download cache");
    let archive = archives.join(name);
    if !archive.is_file() {
        println!("cargo:warning=Downloading {url}");
        // Publish only a complete download. Interrupted attempts can be retried.
        let temporary = archive.with_extension("download");
        let mut response = ureq::get(url).call().expect("download build dependency");
        io::copy(
            &mut response.body_mut().as_reader(),
            &mut File::create(&temporary).unwrap(),
        )
        .expect("save build dependency");
        fs::rename(temporary, &archive).expect("publish build dependency");
    }
    archive
}

fn unpack(archive: &Path, destination: &Path, members: &[&str]) {
    fs::create_dir_all(destination).expect("create source directory");
    run(Command::new("tar")
        .arg("-xf")
        .arg(archive)
        .arg("--strip-components=1")
        .arg("-C")
        .arg(destination)
        .args(members));
}

fn run(command: &mut Command) {
    let output = command.output().expect("start native build command");
    assert!(
        output.status.success(),
        "{command:?} failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
