# bytesetscan

Byte-set membership and search using Shufti, Truffle, and Bitset.

The needle is a set of byte values. Build a matcher once and reuse it to find
delimiters, validate an alphabet, or locate token boundaries. For example,
`b"abc"` matches any occurrence of `a`, `b`, or `c`, rather than the substring
`abc`.

SIMD support is always enabled. The default `std` feature
enables runtime CPU detection on x86, x86_64, and AArch64. For `no_std` builds,
disable default features:

```toml
bytesetscan = { version = "0.0.1", default-features = false }
```

The minimum supported Rust version is 1.89.

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
the positions of bytes outside it. Both methods are available on all three matchers and their explicit engines. Use `.rev()` to iterate backwards, or alternate between
`next()` and `next_back()`:

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
| `Bitset::new(needle)`                     | Builds an exact matcher for any byte set.                                    |
| `Bitset::with_case(ignore_case, needle)`  | Builds a matcher with configurable ASCII case handling for any byte set.     |
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
| `needle()`             | Iterates over the needle in ascending order.                             |

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

## How Shufti works

Shufti encodes a byte set in two tables with 16 entries each. A byte has two
four-bit halves, called **nibbles**, each with a value from 0 to 15:

```text
byte b = (high nibble << 4) | low nibble

low nibble  = b & 0b00001111
high nibble = b >> 4
```

Each table entry is a set of group flags. A byte belongs to the set when the
entries selected by its two nibbles share at least one flag. For `b"ABRabr"`,
the tables can be written as:

```rust
let mut low = [0u8; 16];
let mut high = [0u8; 16];
low[1] = 0b00000001;
low[2] = 0b00000011;
high[4] = 0b00000001;
high[5] = 0b00000010;
high[6] = 0b00000001;
high[7] = 0b00000010;

let accepts = |byte: u8| {
    low[(byte & 0b00001111) as usize] & high[(byte >> 4) as usize] != 0
};

assert!(accepts(b'B'));
assert!(accepts(b'R'));
assert!(!accepts(b'Q'));
```

The membership test uses two table lookups and a bitwise AND. The constructors
build these tables from the needle; the following example explains how.

### Building the groups: `ABRabr`

Imagine a 16-by-16 grid of all possible byte values. The high nibble selects a
row, and the low nibble selects a column. Mark the cells for the bytes to accept.
For `b"ABRabr"`, the nonempty rows are:

| High nibble | Accepted bytes             | Accepted low nibbles | Group flag   |
| ----------- | -------------------------- | -------------------- | ------------ |
| `0x4`       | `A` (`0x41`), `B` (`0x42`) | `{1, 2}`             | `0b00000001` |
| `0x5`       | `R` (`0x52`)               | `{2}`                | `0b00000010` |
| `0x6`       | `a` (`0x61`), `b` (`0x62`) | `{1, 2}`             | `0b00000001` |
| `0x7`       | `r` (`0x72`)               | `{2}`                | `0b00000010` |

The constructor groups rows with identical sets of accepted columns. Rows
`0x4` and `0x6` share one flag because both accept `{1, 2}`. Rows `0x5` and
`0x7` share another because both accept only `{2}`. Empty rows need no flag.

The high table selects a row's group. The low table records every group that
accepts a particular column. The nonzero entries in this example are:

| Table entries        | Flags        | Meaning                                    |
| -------------------- | ------------ | ------------------------------------------ |
| `high[4]`, `high[6]` | `0b00000001` | These rows accept low nibbles 1 and 2.     |
| `high[5]`, `high[7]` | `0b00000010` | These rows accept only low nibble 2.       |
| `low[1]`             | `0b00000001` | Only the first group accepts low nibble 1. |
| `low[2]`             | `0b00000011` | Both groups accept low nibble 2.           |

All other entries are zero. The flags now distinguish bytes that share just
one nibble:

| Byte         | Lookup                             | Result   |
| ------------ | ---------------------------------- | -------- |
| `B` (`0x42`) | `low[2] & high[4]` = `0b11 & 0b01` | Match    |
| `R` (`0x52`) | `low[2] & high[5]` = `0b11 & 0b10` | Match    |
| `Q` (`0x51`) | `low[1] & high[5]` = `0b01 & 0b10` | No match |

Simply checking that both nibbles occur somewhere in the set would incorrectly
accept `Q`: its high nibble occurs in `R`, and its low nibble occurs in `A`.
The shared-flag test requires the two nibbles to belong to the same group.
Since rows share a group only when their accepted columns are identical, the
test accepts exactly the marked cells. No second verification step is needed.

### Choosing a matcher

`Shufti`, `Truffle`, and `Bitset` each occupy 32 bytes.
`Shufti::new` returns `Option<Self>` because
some byte sets cannot be represented by Shufti's eight group flags.
`Truffle::new` and `Bitset::new` return `Self` and accept any byte set.

The Shufti limit counts **distinct nonempty row patterns**, not bytes in the set:

- Any set containing only ASCII bytes fits, since it occupies at most eight
  rows. This also holds after ASCII case expansion.
- Even the set of all 256 bytes fits: every row accepts every column, so all
  rows share a single group.
- Nine bytes can exceed the limit if they occupy nine rows with different
  column patterns, as in the example below.

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

### Why the tables suit SIMD

A SIMD byte shuffle can look up several entries in a 16-byte table at once.
The Shufti table evaluates several bytes in parallel:

1. Extract each byte's low and high nibbles.
2. Shuffle the low and high tables to obtain the corresponding group flags.
3. AND the flags and compare with zero to obtain a mask of matching lanes.
4. Use the mask to locate the first matching byte.

`mismatch` locates the first nonmatching byte instead. `rfind` and `rmismatch`
scan in the opposite direction to locate the last matching or nonmatching byte.
All four operations use the same matching rules with or without SIMD.
Hyperscan's [Shufti implementation](https://github.com/intel/hyperscan/blob/master/src/nfa/shufti.h)
is an example of using byte shuffles for this algorithm.

Whether SIMD improves performance depends on the processor, input length, and
match positions. Measure representative inputs for your application.

## How Truffle works

Truffle represents every possible byte set in 32 bytes, with no grouping limit.
It divides the set into two 16-byte masks: one for values below `128` and one
for values from `128` through `255`.

For a byte `b`, its top bit selects the mask. The low nibble selects an entry,
and bits 4 through 6 select a bit within that entry. A set bit means a match.
For the needle `[0x41, 0x61, 0xC1]`, only two entries are nonzero:

| Entry      | Mask         | Accepted bytes |
| ---------- | ------------ | -------------- |
| `lower[1]` | `0b01010000` | `0x41`, `0x61` |
| `upper[1]` | `0b00010000` | `0xC1`         |

For example, `0xC1` selects `upper[1]` and tests bit 4, which is set. `0x81`
selects the same entry but tests bit 0, which is clear. Each of the 256 possible
byte values has its own bit, so arbitrary sets are represented exactly.

The SIMD search uses three byte shuffles: two to look up the mask entries and
one to form the bit to test. It selects the appropriate mask according to each
input byte's top bit and tests the selected bit. The algorithm is described in
Hyperscan's [table construction](https://github.com/intel/hyperscan/blob/master/src/nfa/trufflecompile.cpp)
and [Truffle search](https://github.com/intel/hyperscan/blob/master/src/nfa/truffle.c).

## How Bitset works

Bitset assigns one bit to each possible byte value. Four `u64` words cover all
256 values in 32 bytes. For a byte `b`, `b / 64` selects the word and `b % 64`
selects the bit to test. Inserting a byte sets that bit; duplicates leave the
set unchanged. No grouping or table-construction failure is needed.

For example, `0x41` (65) selects word 1, bit 1; `0xC1` (193) selects word 3,
bit 1. These positions are independent, so either byte can be present without
the other. A search checks bytes against this bitset and stops at the
first result in the requested direction.

For SIMD lookup, the same bits form a 32-byte table: `b >> 3` selects a
byte, and `1 << (b & 0b00000111)` selects its bit. This lets each SIMD lane
test membership directly in the bitset.

```rust
use bytesetscan::bitset::Bitset;

let bytes = Bitset::new(&[0x00, 0x41, 0xC1, 0xFF]);
assert_eq!(bytes.find(&[0x42, 0xC1, 0xFF]), Some(1));
assert_eq!(bytes.rfind(&[0x42, 0xC1, 0xFF]), Some(2));
assert_eq!(bytes.mismatch(&[0x41, 0x42, 0xC1]), Some(1));
```

| Matcher   | Table storage | Byte sets accepted              | Search                    |
| --------- | ------------- | ------------------------------- | ------------------------- |
| `Shufti`  | 32 bytes      | Up to eight distinct row groups | SIMD with scalar fallback |
| `Truffle` | 32 bytes      | Any byte set                    | SIMD with scalar fallback |
| `Bitset`  | 32 bytes      | Any byte set                    | SIMD with scalar fallback |

Shufti uses fewer table lookups per SIMD block, while Truffle accepts arbitrary
byte sets. Bitset gives each possible byte its own membership bit. Compare representative workloads on the processors you intend to
support; these differences alone do not establish a performance ranking.

## Inspecting the needle

`needle()` iterates over the needle represented by a matcher: the
matching byte set, not the original `needle` supplied to the constructor.
Bytes are yielded in ascending order without duplicates.

```rust
use bytesetscan::shufti::Shufti;

let exact = Shufti::new(b"baba").unwrap();
assert!(exact.needle().eq(b"ab".iter().copied()));

let folded = Shufti::with_case(true, b"bA").unwrap();
assert!(folded.needle().eq(b"ABab".iter().copied()));
```

For ASCII case-insensitive matchers, the needle includes both cases of each
letter, regardless of the constructor's input.

## SIMD support

Shufti, Truffle, and Bitset select a supported SIMD implementation automatically:

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

Use `ShuftiEngine`, `TruffleEngine`, or `BitsetEngine` to specify a backend.
Each is available in its algorithm's module, alongside the automatic matcher.

`ShuftiEngine::new(backend, needle)` returns `None` if the backend is unavailable
or the needle cannot be represented; see [choosing a matcher](#choosing-a-matcher)
for the accepted byte sets. `TruffleEngine::new(backend, needle)` and
`BitsetEngine::new(backend, needle)` accept every byte set and return `None`
only when the backend is unavailable.

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

## Costs and reuse

Building a matcher or engine from a needle
takes time proportional to its length, plus a constant setup cost. The result
does not borrow the needle, so the original input can be dropped or changed.

Searches take linear time in the haystack length in the worst case. Reuse a
matcher across slices when the needle stays the same to avoid repeating
construction.

## Benchmarks

The [scan benchmark](benches/README.md) compares Shufti, Truffle, and Bitset
`find` and `rfind` with Hyperscan's Shufti and Truffle implementations.
It checks results before timing and reports median nanoseconds per search with
the observed minimum and maximum:

```sh
cargo bench --bench scan
```

Use `cargo bench --bench scan -- --help` for inputs, backends, and measurement
time budgets. The default backend follows `SimdBackend::new()`; unsupported
implementations display `—`. Table construction is excluded from timing.
The separate benchmark workspace builds Hyperscan on Linux x86-64;
ordinary library builds and tests do not require it.

## Checking the documentation

The README is also the crate-level API documentation. Run its Rust examples and
the method examples with each feature configuration:

```sh
cargo test --doc
cargo test --doc --no-default-features
```
