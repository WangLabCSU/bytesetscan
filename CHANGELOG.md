# Changelog

## 0.0.1

Initial release.

- Shufti matching for byte sets representable by up to eight distinct row groups.
- Truffle matching for arbitrary byte sets.
- Forward and reverse searches for matching and nonmatching bytes.
- Double-ended iterators over matching and nonmatching positions.
- Optional ASCII case-insensitive matching.
- Automatic backend selection and explicitly configured matchers.
- SIMD implementations for x86/x86-64, AArch64 NEON, and WebAssembly SIMD128,
  with a scalar fallback.
- `no_std` support by disabling default features; no runtime dependencies.

Requires Rust 1.89 or later.
