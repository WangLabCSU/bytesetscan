use std::{
    fs::File,
    hint::black_box,
    io::{BufWriter, Write},
    path::PathBuf,
    time::{Duration, Instant},
};

use bytesetscan::{shufti::ShuftiEngine, truffle::TruffleEngine, SimdBackend};
use clap::Parser;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
use hyperscan::{Algorithm, Backend, Matcher};

mod input;
mod report;
mod sampling;

use self::input::{Hit, Input};
use self::sampling::Sample;

#[derive(Parser)]
#[command(about = "Compare byte-set find/rfind across available backends")]
struct Options {
    /// Haystack lengths in bytes; may be repeated or comma-separated.
    #[arg(long, value_delimiter = ',', default_value = "1,7,15,64,1024")]
    size: Vec<usize>,
    /// Number of distinct needle bytes, from 1 through 255.
    #[arg(long, value_delimiter = ',', default_value = "4,16,64", value_parser = clap::value_parser!(u16).range(1..=255))]
    needle_size: Vec<u16>,
    /// Match placement in the haystack.
    #[arg(long, value_delimiter = ',', value_enum, default_value = "middle,miss")]
    hit: Vec<Hit>,
    /// auto selects the machine's preferred backend; all tests every available backend.
    #[arg(long, default_value = "auto", value_parser = ["auto", "all", "fallback", "ssse3", "sse42", "avx2", "avx512", "vbmi", "neon", "simd128"])]
    backend: String,
    /// Measured rounds; implementations take turns running first.
    #[arg(long, default_value_t = 10, value_parser = clap::value_parser!(u32).range(1..))]
    rounds: u32,
    /// Minimum milliseconds per measured batch, with automatic iteration counts.
    #[arg(long, default_value_t = 10, value_parser = clap::value_parser!(u64).range(1..))]
    sample_ms: u64,
    /// Warmup milliseconds for each method and input before measurement.
    #[arg(long, default_value_t = 30, value_parser = clap::value_parser!(u64).range(1..))]
    warmup_ms: u64,
    /// Use a fixed search count instead of the automatic batch duration.
    #[arg(long, conflicts_with = "sample_ms", value_parser = clap::value_parser!(u32).range(1..))]
    iterations: Option<u32>,
    /// Save every measured batch's iteration count and elapsed time as CSV.
    #[arg(long, conflicts_with = "test")]
    samples: Option<PathBuf>,
    /// Haystack address modulo 64, from 0 through 63.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u8).range(0..64))]
    offset: u8,
    /// Check every available implementation without measuring time.
    #[arg(long)]
    test: bool,
}

fn main() {
    if let Err(error) = run(Options::parse()) {
        eprintln!("benchmark failed: {error}");
        std::process::exit(1);
    }
}

fn run(options: Options) -> Result<(), String> {
    let backends = selected_backends(&options.backend)?;
    let mut output = options
        .samples
        .as_ref()
        .map(File::create)
        .transpose()
        .map_err(|error| format!("create samples file: {error}"))?
        .map(BufWriter::new);
    if let Some(writer) = &mut output {
        writeln!(
            writer,
            "backend,needle,haystack,hit,search,method,round,iterations,elapsed_ns,ns_per_search"
        )
        .map_err(|error| error.to_string())?;
    }
    if !options.test {
        println!("Times are median [min, max] ns/search across measured batches.");
        println!("Green marks the faster implementation in each row; ★ precedes the fastest measurement in each search group.");
        println!(
            "Equal displayed medians share markers; markers do not imply statistical significance."
        );
        println!("— means the implementation cannot use this backend or needle.");
        if let Some(iterations) = options.iterations {
            println!(
                "Warmup >= {} ms/method; {} rounds of {iterations} searches (fixed count).",
                options.warmup_ms, options.rounds
            );
        } else {
            println!(
                "Warmup >= {} ms/method; {} rounds; each recorded batch >= {} ms.",
                options.warmup_ms, options.rounds, options.sample_ms
            );
        }
    }
    let mut verified = 0;
    for backend in backends {
        if options.test {
            println!("Checking {}", backend_name(backend));
        } else {
            report::header(backend_name(backend));
        }
        for &needle_size in &options.needle_size {
            let needle: Vec<_> = (0..needle_size).map(|byte| byte as u8).collect();
            let methods = prepare(backend, &needle);
            for &len in &options.size {
                for &hit in &options.hit {
                    let input = Input::new(len, hit, options.offset as usize);
                    // Construction gives an explicit expected position. Missing
                    // implementations are omitted, never replaced by a fallback.
                    for method in methods.iter().flatten() {
                        assert_eq!(
                            method.result(&input, false),
                            input.expected(),
                            "{} find",
                            method.name()
                        );
                        assert_eq!(
                            method.result(&input, true),
                            input.expected(),
                            "{} rfind",
                            method.name()
                        );
                        verified += 2;
                    }
                    if options.test {
                        continue;
                    }
                    for reverse in [false, true] {
                        let samples = measure_case(&methods, &input, reverse, &options);
                        report::rows(
                            needle_size,
                            len,
                            hit,
                            reverse,
                            samples.each_ref().map(|samples| report::summarize(samples)),
                        );
                        if let Some(writer) = &mut output {
                            // Save after the entire comparison, outside timed loops.
                            report::write_samples(
                                writer,
                                backend_name(backend),
                                needle_size,
                                len,
                                hit,
                                reverse,
                                &samples,
                            )
                            .map_err(|error| format!("write samples: {error}"))?;
                        }
                    }
                }
            }
        }
    }
    if let Some(writer) = &mut output {
        writer
            .flush()
            .map_err(|error| format!("flush samples: {error}"))?;
    }
    println!("Verified {verified} find/rfind results before timing.");
    Ok(())
}

fn selected_backends(requested: &str) -> Result<Vec<SimdBackend>, String> {
    let available: Vec<_> = SimdBackend::backends()
        .filter(SimdBackend::is_available)
        .collect();
    let selected = match requested {
        "auto" => vec![SimdBackend::new()],
        "all" => available.clone(),
        name => available
            .iter()
            .copied()
            .filter(|&backend| backend_name(backend) == name)
            .collect(),
    };
    if selected.is_empty() {
        let names = available
            .into_iter()
            .map(backend_name)
            .collect::<Vec<_>>()
            .join(", ");
        Err(format!(
            "backend {requested} is unavailable; this machine supports: {names}"
        ))
    } else {
        Ok(selected)
    }
}

fn backend_name(backend: SimdBackend) -> &'static str {
    match backend {
        SimdBackend::Fallback => "fallback",
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Ssse3 => "ssse3",
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Sse4_2 => "sse42",
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Avx2 => "avx2",
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Avx512 => "avx512",
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Avx512Vbmi => "vbmi",
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => "neon",
        #[cfg(target_arch = "wasm32")]
        SimdBackend::WasmSimd128 => "simd128",
    }
}

fn prepare(backend: SimdBackend, needle: &[u8]) -> [Option<Prepared>; 4] {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    let native = hyperscan_backend(backend)
        .map(|backend| {
            [
                Matcher::new(backend, Algorithm::Shufti, needle).map(Prepared::HsShufti),
                Matcher::new(backend, Algorithm::Truffle, needle).map(Prepared::HsTruffle),
            ]
        })
        .unwrap_or([None, None]);
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    let native = [None, None];
    let [hs_shufti, hs_truffle] = native;
    [
        ShuftiEngine::new(backend, needle).map(Prepared::Shufti),
        TruffleEngine::new(backend, needle).map(Prepared::Truffle),
        hs_shufti,
        hs_truffle,
    ]
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn hyperscan_backend(backend: SimdBackend) -> Option<Backend> {
    match backend {
        SimdBackend::Fallback => None,
        SimdBackend::Ssse3 => Some(Backend::Ssse3),
        SimdBackend::Sse4_2 => Some(Backend::Sse42),
        SimdBackend::Avx2 => Some(Backend::Avx2),
        SimdBackend::Avx512 => Some(Backend::Avx512),
        SimdBackend::Avx512Vbmi => Some(Backend::Avx512Vbmi),
    }
}

fn measure_case(
    methods: &[Option<Prepared>; 4],
    input: &Input,
    reverse: bool,
    options: &Options,
) -> [Vec<Sample>; 4] {
    let active: Vec<_> = methods
        .iter()
        .enumerate()
        .filter_map(|(index, method)| method.as_ref().map(|method| (index, method)))
        .collect();
    let sample_time = Duration::from_millis(options.sample_ms);
    let warmup_time = Duration::from_millis(options.warmup_ms);
    let mut counts = [0; 4];
    for &(index, method) in &active {
        let estimated = sampling::warmup(
            |count| method.measure(input, reverse, count),
            warmup_time,
            sample_time,
        );
        counts[index] = options.iterations.map_or(estimated, |count| count as usize);
    }

    let mut samples = std::array::from_fn(|_| Vec::new());
    for round in 0..options.rounds as usize {
        for slot in 0..active.len() {
            // Rotate only the methods actually being measured. Missing columns
            // must not give one available implementation extra first runs.
            let (index, method) = active[(round + slot) % active.len()];
            let sample = sampling::sample(
                |count| method.measure(input, reverse, count),
                &mut counts[index],
                options.iterations.is_none().then_some(sample_time),
            );
            samples[index].push(sample);
        }
    }
    samples
}

enum Prepared {
    Shufti(ShuftiEngine),
    Truffle(TruffleEngine),
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    HsShufti(Matcher),
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    HsTruffle(Matcher),
}

impl Prepared {
    fn name(&self) -> &'static str {
        match self {
            Self::Shufti(_) => "Shufti",
            Self::Truffle(_) => "Truffle",
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
            Self::HsShufti(_) => "Hyperscan Shufti",
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
            Self::HsTruffle(_) => "Hyperscan Truffle",
        }
    }

    fn result(&self, input: &Input, reverse: bool) -> Option<usize> {
        let haystack = input.haystack();
        match self {
            Self::Shufti(m) => {
                if reverse {
                    m.rfind(haystack)
                } else {
                    m.find(haystack)
                }
            }
            Self::Truffle(m) => {
                if reverse {
                    m.rfind(haystack)
                } else {
                    m.find(haystack)
                }
            }
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
            Self::HsShufti(m) | Self::HsTruffle(m) => {
                if reverse {
                    // SAFETY: Input retains prefix storage in the same allocation.
                    unsafe { m.rfind(haystack) }
                } else {
                    m.find(haystack)
                }
            }
        }
    }

    fn measure(&self, input: &Input, reverse: bool, iterations: usize) -> Duration {
        // Choose the implementation and direction before starting the timer.
        let haystack = input.haystack();
        match black_box(self) {
            Self::Shufti(m) => {
                measure_direction(|h| m.find(h), |h| m.rfind(h), haystack, reverse, iterations)
            }
            Self::Truffle(m) => {
                measure_direction(|h| m.find(h), |h| m.rfind(h), haystack, reverse, iterations)
            }
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
            Self::HsShufti(m) | Self::HsTruffle(m) => measure_direction(
                |h| m.find(h),
                // SAFETY: only Input's slice, with its allocated prefix, is passed.
                |h| unsafe { m.rfind(h) },
                haystack,
                reverse,
                iterations,
            ),
        }
    }
}

fn measure_direction(
    find: impl Fn(&[u8]) -> Option<usize>,
    rfind: impl Fn(&[u8]) -> Option<usize>,
    haystack: &[u8],
    reverse: bool,
    iterations: usize,
) -> Duration {
    if reverse {
        measure(rfind, haystack, iterations)
    } else {
        measure(find, haystack, iterations)
    }
}

fn measure(
    search: impl Fn(&[u8]) -> Option<usize>,
    haystack: &[u8],
    iterations: usize,
) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(search(black_box(haystack)));
    }
    start.elapsed()
}

#[cfg(test)]
mod tests {
    use super::{prepare, selected_backends, SimdBackend};

    #[test]
    fn auto_uses_the_librarys_backend_selection() {
        assert_eq!(selected_backends("auto").unwrap(), [SimdBackend::new()]);
    }

    #[test]
    fn all_includes_every_available_backend() {
        let expected: Vec<_> = SimdBackend::backends()
            .filter(SimdBackend::is_available)
            .collect();
        assert_eq!(selected_backends("all").unwrap(), expected);
    }

    #[test]
    fn prepare_leaves_hyperscan_missing_for_fallback() {
        let methods = prepare(SimdBackend::Fallback, b"a");
        assert!(methods[2].is_none());
        assert!(methods[3].is_none());
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn prepare_keeps_rust_neon_without_hyperscan() {
        if !SimdBackend::Neon.is_available() {
            return;
        }
        let methods = prepare(SimdBackend::Neon, b"a");
        assert!(methods[..2].iter().all(Option::is_some));
        assert!(methods[2..].iter().all(Option::is_none));
    }
}
