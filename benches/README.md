# Scan benchmark

Compare the public Shufti and Truffle engine interfaces with the
original Hyperscan Shufti and Truffle searches. The runner verifies results,
warms up, rotates method order, and prints a comparison table.
Each backend has its own table: algorithms are rows,
needle size, haystack length, and hit placement are columns alongside the
bytesetscan and Hyperscan results. Find/rfind form separate groups.
Each cell shows the median and observed minimum/maximum in
nanoseconds per search. Green marks the faster implementation in each algorithm
row. A `★` before a measurement marks the fastest result across all algorithms
and both libraries for the same backend, input, and search direction. Equal displayed
medians share markers. Neither marker implies statistical significance;
overlapping ranges need further measurement. Redirected output and `NO_COLOR`
disable color, while stars remain visible. Unavailable implementations
display `—` and receive no markers.

```sh
cargo bench --bench scan
cargo bench --bench scan -- --help

# Compare short and long slices on every available backend.
cargo bench --bench scan -- --backend all --size 1,7,15,16,31,32,63,64,1024,16384,65536

# Select needle sizes and match positions.
cargo bench --bench scan -- --needle-size 1,8,16,24,32,48,64,96,128,192,255 --hit miss,first,middle,last

# Verify the selected cases without timing.
cargo bench --bench scan -- --test --backend all --size 0,1,7,15,16,17,31,32,33,63,64,65,4096

# Repeat the 64-byte needle / 1,024-byte haystack comparison and save samples.
cargo bench --bench scan -- --size 1024 --needle-size 64 --hit miss --samples target/scan-samples.csv

# Short exploratory run with an explicit time budget.
cargo bench --bench scan -- --size 1024 --needle-size 16 --rounds 5 --warmup-ms 10 --sample-ms 5

# On AArch64, select NEON explicitly or let the default auto selection choose it.
cargo bench --bench scan -- --backend neon
```

| Option          | Default                    | Meaning                                                                                       |
| --------------- | -------------------------- | --------------------------------------------------------------------------------------------- |
| `--size`        | `7,15,64,128,256,512,1024` | Haystack lengths in bytes.                                                                    |
| `--needle-size` | `4,16,64`                  | Distinct needle bytes, from 1 through 255.                                                    |
| `--hit`         | `middle,miss`              | `miss`, `first`, `middle`, or `last` match placement.                                         |
| `--backend`     | `auto`                     | Automatic selection, `all`, or a backend from the table below.                                |
| `--rounds`      | `10`                       | Recorded batches per method and search case.                                                  |
| `--warmup-ms`   | `30`                       | Minimum warmup time per method and case.                                                      |
| `--sample-ms`   | `10`                       | Minimum duration of each recorded batch.                                                      |
| `--iterations`  | Automatic                  | Override the batch duration with a fixed search count; conflicts with explicit `--sample-ms`. |
| `--samples`     | Off                        | CSV path for every recorded batch's count and elapsed time.                                   |
| `--offset`      | `1`                        | Haystack address modulo 64, from 0 through 63.                                                |
| `--test`        | Off                        | Check results without timing.                                                                 |

Sizes, needle sizes, and hit positions accept comma-separated values or
repeated options. Each needle contains bytes `0..size`; each haystack contains
byte 255, with one byte replaced by zero for a hit. This makes the expected
result explicit and keeps all sets representable by every implementation.
These controlled inputs measure scan length and table density; they do not
represent every possible byte-set layout or input distribution.

`auto` uses `SimdBackend::new()` on the current machine. `all` runs every
available backend, including Fallback. Explicitly requesting an unavailable
backend produces an error and lists the available choices.

| Backend    | Rust implementations          | Hyperscan reference          |
| ---------- | ----------------------------- | ---------------------------- |
| `fallback` | All targets                   | —                            |
| `ssse3`    | x86 and x86-64                | Linux x86-64, when supported |
| `sse42`    | x86 and x86-64                | Linux x86-64, when supported |
| `avx2`     | x86 and x86-64                | Linux x86-64, when supported |
| `avx512`   | x86 and x86-64                | Linux x86-64, when supported |
| `vbmi`     | x86 and x86-64                | Linux x86-64, when supported |
| `neon`     | AArch64                       | —                            |
| `simd128`  | Wasm32 with `simd128` enabled | —                            |

No implementation silently substitutes another backend. A missing backend,
unsupported native platform, or unrepresentable needle leaves its cell as `—`.
The current controlled needles fit both Rust algorithms. Hyperscan's
upstream CPU targets can require additional features, so its cell can also be
missing when the corresponding Rust backend is available.

## Measurement

Each method warms up with increasing batch sizes for at least `--warmup-ms`.
The measured rate determines its iteration count for `--sample-ms`. If a later
batch finishes too quickly, its count is increased and that short calibration
batch is retried. Every recorded automatic batch therefore reaches the requested
duration. Slow samples are retained, including interrupted ones. Methods rotate
their running order between rounds; their iteration counts can differ.

Statistics use each batch's elapsed time divided by its own iteration count.
The displayed range is the minimum and maximum of those batch averages, not
a confidence interval or individual-call latency range. `--samples` records
the backend, input, operation, method, round, iterations, elapsed nanoseconds,
and normalized time. Missing methods have no sample rows. CSV writes take place
after each case's measurements.

Time budgets and explicit ranges make noise easier to detect; they do not remove
CPU frequency changes, operating-system scheduling, or virtual-machine pauses.
For stronger conclusions, repeat the same filtered cases on a quiet machine
and compare ranges across runs. Fixed `--iterations` bypasses the minimum batch
duration and is intended for controlled experiments.

The buffer allocation includes enough padding to fix the actual input address
modulo 64 to `--offset`, independently of the allocator's returned alignment.

Each row uses the same slice and backend for all implementations. `find` and
`rfind` use the same input, so a hit at the last byte is immediately found by
`rfind`. Times are median **nanoseconds per search**, not throughput over bytes
that an early return did not inspect. Construction, input allocation, result
verification, and choosing the measured method happen before timing. The timer
includes the loop, `black_box`, and the search call. Hyperscan calls include its
small C ABI wrapper and stored kernel pointer; Rust calls include the engine's
stored kernel pointer. These are interface comparisons, not isolated instruction
timings. No mismatch operation is synthesized for Hyperscan.

## Native build

Only the Hyperscan reference requires Linux x86-64, a C/C++ compiler, CMake,
Make, binutils, and tar. On other platforms the Rust comparisons still run,
with `—` in the Hyperscan column; the native build is skipped. On Linux x86-64,
the first run downloads missing sources using Rust's `ureq`
client and prepares them under `target/scan`. Boost headers and Ragel are reused
from the system when available, otherwise prepared in that cache. Nothing is
installed system-wide.

The pinned [Hyperscan](https://github.com/intel/hyperscan) revision is
`bc3b191ab56055e8560c7cdc161c289c4d76e3d2`.
Its original CMake `hs_runtime` target is built in Release mode with
`FAT_RUNTIME`, `BUILD_AVX512`, and `BUILD_AVX512VBMI` enabled. Upstream chooses
the compiler flags for each backend. The wrapper compiles the two upstream
table builders separately; neither search source is modified. An installed
public Hyperscan shared library is insufficient because the benchmark calls
internal byte-set kernels.

To reuse local files, set these to absolute paths before invoking Cargo:

| Variable           | Meaning                                                                                                                               |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| `HYPERSCAN_SOURCE` | Original Hyperscan source directory.                                                                                                  |
| `HYPERSCAN_BUILD`  | Matching CMake build directory containing `lib/libhs_runtime.a` and generated headers; all five fat-runtime backends must be enabled. |
| `BOOST_ROOT`       | Boost root containing `boost/` or `include/boost/`.                                                                                   |
| `RAGEL`            | Ragel executable path.                                                                                                                |

Original reverse kernels form a pointer one byte before the searched slice.
The input owns prefix storage in the same allocation to satisfy this requirement;
the wrapper documents it on its unsafe `rfind` method. Both libraries search
the identical subslice, without copying it during timing.

## Files

- `scan/main.rs`: Cargo benchmark launcher.
- `scan/src/main.rs`: CLI, engine preparation, verification, and timing.
- `scan/src/input.rs`: Input construction and expected positions.
- `scan/src/report.rs`: Medians and table output.
- `scan/src/sampling.rs`: Warmup, batch duration calibration, and raw samples.
- `scan/hyperscan/`: Native build and a wrapper exposing construction and search.

The nested workspace keeps native dependencies out of ordinary library tests.
Run its tests with
`cargo test --release --workspace --manifest-path benches/scan/Cargo.toml --target-dir target/scan`.

For Wasm SIMD128, build the runner directly from the nested workspace with
`--target wasm32-wasip1` and `RUSTFLAGS='-C target-feature=+simd128'`, and use
a configured WASI runner. The outer Cargo launcher starts a subprocess, which
is intended for native platforms. Cross-compilation checks do not establish
runtime correctness or performance on the target machine.
