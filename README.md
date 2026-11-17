# bytesetscan

[<img alt="github" src="https://img.shields.io/badge/github-WangLabCSU/bytesetscan-8da0cb?style=for-the-badge&labelColor=555555&logo=github" height="20">](https://github.com/WangLabCSU/bytesetscan)
[<img alt="crates.io" src="https://img.shields.io/crates/v/bytesetscan.svg?style=for-the-badge&color=fc8d62&logo=rust" height="20">](https://crates.io/crates/bytesetscan)
[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-bytesetscan-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs" height="20">](https://docs.rs/bytesetscan)
[<img alt="build status" src="https://img.shields.io/github/actions/workflow/status/WangLabCSU/bytesetscan/ci.yml?branch=main&style=for-the-badge" height="20">](https://github.com/WangLabCSU/bytesetscan/actions/workflows/ci.yml)

Byte-set membership and search using Shufti and Truffle.

Shufti uses two nibble-indexed tables of group flags; Truffle uses a 16-by-16
bitmap with one bit for each possible byte. Both algorithms are used in
[Hyperscan](https://github.com/intel/hyperscan). Wojciech Muła's
[*SIMDized check which bytes are in a set*](http://0x80.pl/notesen/2018-10-18-simd-byte-lookup.html)
(2018) explains the corresponding SIMD lookup techniques in “Special case 1 —
small sets” and “Universal algorithm”.

The needle is a set of byte values. Build a matcher once and reuse it to find
delimiters, validate an alphabet, or locate token boundaries. For example,
`b"abc"` matches any occurrence of `a`, `b`, or `c`, rather than the substring
`abc`.

## Quick start

Construct a Shufti matcher directly from the needle:

```rust
use bytesetscan::shufti::Shufti;

let whitespace = Shufti::new(b" \t\r\n").unwrap();
assert_eq!(whitespace.find(b"hello world"), Some(5));
assert_eq!(whitespace.find(b"hello"), None);
assert_eq!(whitespace.rfind(b"one two three"), Some(7));

let bases = Shufti::with_case(true, b"ACGTU").unwrap();
assert_eq!(bases.mismatch(b"acGU?"), Some(4));
assert_eq!(bases.mismatch(b"acGU"), None);
assert_eq!(bases.rmismatch(b"?acGU!ac"), Some(5));
```

`Truffle` provides the same searches and iterators and accepts any byte set.

```rust
use bytesetscan::truffle::Truffle;

let bytes = Truffle::new(&[0, 127, 128, 255]);
assert_eq!(bytes.find(&[1, 2, 255]), Some(2));
assert_eq!(bytes.mismatch(&[0, 128, 7]), Some(2));
```

`finder()` yields the positions of bytes in the needle; `mismatcher()` yields
the positions of bytes outside it.

```rust
use bytesetscan::shufti::Shufti;

let whitespace = Shufti::new(b" \t").unwrap();
assert!(whitespace.finder(b"a b\tc").eq([1, 3]));
assert!(whitespace.finder(b"a b\tc").rev().eq([3, 1]));
assert!(whitespace.mismatcher(b"a b\tc").eq([0, 2, 4]));
```

| Constructor                               | Result                                                                       |
| ----------------------------------------- | ---------------------------------------------------------------------------- |
| `Shufti::new(needle)`                     | Builds an exact matcher, or returns `None` if the set cannot be represented. |
| `Shufti::with_case(ignore_case, needle)`  | Builds a matcher with configurable ASCII case handling, with the same limit. |
| `Truffle::new(needle)`                    | Builds an exact matcher for any byte set.                                    |
| `Truffle::with_case(ignore_case, needle)` | Builds a matcher with configurable ASCII case handling for any byte set.     |

| Matcher operation      | Result                                                                   |
| ---------------------- | ------------------------------------------------------------------------ |
| `find(haystack)`       | Index of the first byte in the needle, or `None` if no match is found.   |
| `mismatch(haystack)`   | Index of the first byte not in the needle, or `None` if all bytes match. |
| `rfind(haystack)`      | Index of the last byte in the needle, or `None` if no match is found.    |
| `rmismatch(haystack)`  | Index of the last byte not in the needle, or `None` if all bytes match.  |
| `finder(haystack)`     | Iterates over the indices of bytes in the needle.                        |
| `mismatcher(haystack)` | Iterates over the indices of bytes outside the needle.                   |

`rfind` and `rmismatch` search from the end of the slice and return positions
within the original input. All four operations return `None` for empty input.

An empty needle matches no bytes: `find` and `rfind` always return `None`.
For nonempty input, `mismatch` returns `Some(0)`, and `rmismatch` returns the
last byte's index. Duplicate bytes and input order do not affect the needle.

All values from `0x00` through `0xFF` are supported. Inputs are byte slices;
matching does not decode UTF-8. `with_case(true, needle)` accepts either case of
ASCII letters; `with_case(false, needle)` matches exactly, like `new(needle)`.
Nonletters and non-ASCII bytes are always matched exactly. Matching does not perform
Unicode case folding or convert the haystack's case during searches.

## Choosing a matcher

`Shufti` and `Truffle` each occupy 32 bytes.
`Shufti::new` returns `Option<Self>` because its grouping strategy cannot
represent every byte set within eight groups.
`Truffle::new` returns `Self` and accepts any byte set.

The Shufti constructors accept up to eight **distinct nonempty row patterns**.
A row contains bytes with the same high nibble; its pattern is their set of low
nibbles.

Use Shufti when the needle fits, or Truffle when arbitrary sets must be
accepted without construction failure:

```rust
use bytesetscan::{shufti::Shufti, truffle::Truffle};

// Each byte has a different high nibble and a different low nibble.
let needle = [0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
assert!(Shufti::new(&needle).is_none());

let bytes = Truffle::new(&needle);
assert_eq!(bytes.find(&[0x88]), Some(0));
assert_eq!(bytes.find(&[0x89]), None);
```

| Matcher   | Table storage | Byte sets accepted              | Search                    |
| --------- | ------------- | ------------------------------- | ------------------------- |
| `Shufti`  | 32 bytes      | Up to eight distinct row groups | SIMD with scalar fallback |
| `Truffle` | 32 bytes      | Any byte set                    | SIMD with scalar fallback |

Shufti uses fewer table lookups per SIMD block, while Truffle accepts arbitrary
byte sets. Compare representative workloads on the processors you intend to
support; these differences alone do not establish a performance ranking.

## Compiling without the standard library

Disable the default features to use bytesetscan in a `no_std` crate:

```toml
[dependencies]
bytesetscan = { version = "0.0.1", default-features = false }
```

Without `std`, SIMD selection uses the features enabled for the compilation
target. See [SIMD support](#simd-support) for the supported platforms.

## SIMD support

Shufti and Truffle select a supported SIMD implementation automatically:

| Build configuration        | Standard library | Automatic SIMD selection                                           |
| -------------------------- | ---------------- | ------------------------------------------------------------------ |
| Default (`std`)            | Required         | Runtime detection on x86/x86_64/AArch64; target features on wasm32 |
| `default-features = false` | Not required     | Compilation target                                                 |

On AArch64, `std` builds detect NEON support; without `std`, the compilation
target must enable NEON. On wasm32, the compilation target must enable
`simd128` in either build. Other architectures, and targets without supported
SIMD features, use a scalar fallback.

Without runtime detection, searches use the instruction sets enabled for the
compilation target, even if the processor supports additional ones.

The x86 backends support SSSE3, SSE4.2, AVX2, AVX-512F/BW, and AVX-512 VBMI.
Targets without a supported SIMD implementation use scalar matching.
The crate has no third-party runtime dependencies.

### Selecting a backend

Use `ShuftiEngine` or `TruffleEngine` to specify a backend.
Each is available in its algorithm's module, alongside the automatic matcher.

`ShuftiEngine::new(backend, needle)` returns `None` if the backend is unavailable
or the needle cannot be represented; see [choosing a matcher](#choosing-a-matcher)
for the accepted byte sets. `TruffleEngine::new(backend, needle)` accepts every
byte set and returns `None` only when the backend is unavailable.

```rust
use bytesetscan::{shufti::ShuftiEngine, SimdBackend};

let engine = ShuftiEngine::new(SimdBackend::Fallback, b" \t").unwrap();
assert_eq!(engine.find(b"hello world"), Some(5));
assert_eq!(engine.rfind(b"one two\tthree"), Some(7));
assert_eq!(engine.mismatch(b" \tword"), Some(2));
```

`Fallback` is always available. SIMD variants depend on the target architecture;
for example, x86 and x86-64 provide `SimdBackend::Avx2`. Availability follows the
same `std` and target-feature rules as automatic selection.

Use `ShuftiEngine::with_case` to ignore ASCII letter case:

```rust
use bytesetscan::{shufti::ShuftiEngine, SimdBackend};

let engine = ShuftiEngine::with_case(true, SimdBackend::Fallback, b"a").unwrap();
assert_eq!(engine.find(b"!A!"), Some(1));
```
