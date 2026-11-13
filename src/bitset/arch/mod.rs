// Automatic and explicitly configured searches share kernel selection.

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm32;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_64;

use crate::backend::SimdBackend;

// Each function returns haystack.len() when no match is found. Before calling
// a SIMD kernel, establish backend availability; entry points handle all lengths.
// Selecting or storing a pointer does not execute its target instructions.
pub(super) type SearchFn = unsafe fn(&[u64; 4], &[u8]) -> usize;

// Used by automatic dispatch and BitsetEngine construction. Keeping selection
// separate from execution lets an engine retain the kernel without scanning
// any input. Callers must validate the backend before invoking the result.
#[inline(always)]
pub(super) fn search_fn<const REVERT: bool>(backend: SimdBackend) -> SearchFn {
    match backend {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Avx2 => x86_64::bitset_avx2::<REVERT>,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Ssse3 => x86_64::bitset_ssse3::<REVERT>,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Sse4_2 => x86_64::bitset_sse42::<REVERT>,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Avx512 => x86_64::bitset_avx512::<REVERT>,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Avx512Vbmi => x86_64::bitset_vbmi::<REVERT>,
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => aarch64::search::<REVERT>,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        SimdBackend::WasmSimd128 => wasm32::search::<REVERT>,
        _ => search_fallback::<REVERT>,
    }
}

#[inline(always)]
pub(super) fn search_fallback<const REVERT: bool>(bits: &[u64; 4], haystack: &[u8]) -> usize {
    // An empty or exhausted slice uses the same end sentinel as SIMD kernels.
    haystack
        .iter()
        .position(|&byte| contains(bits, byte) != REVERT)
        .unwrap_or(haystack.len())
}

// Reverse scans have their own entry points; membership and nonmembership
// remain the only const-generic choice. As with search_fn, selection alone
// does not establish that the returned SIMD function can be called safely.
#[inline(always)]
pub(super) fn rsearch_fn<const REVERT: bool>(backend: SimdBackend) -> SearchFn {
    match backend {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Avx2 => x86_64::rbitset_avx2::<REVERT>,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Ssse3 => x86_64::rbitset_ssse3::<REVERT>,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Sse4_2 => x86_64::rbitset_sse42::<REVERT>,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Avx512 => x86_64::rbitset_avx512::<REVERT>,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Avx512Vbmi => x86_64::rbitset_vbmi::<REVERT>,
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => aarch64::rsearch::<REVERT>,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        SimdBackend::WasmSimd128 => wasm32::rsearch::<REVERT>,
        _ => rsearch_fallback::<REVERT>,
    }
}

#[inline(always)]
pub(super) fn rsearch_fallback<const REVERT: bool>(bits: &[u64; 4], haystack: &[u8]) -> usize {
    haystack
        .iter()
        .rposition(|&byte| contains(bits, byte) != REVERT)
        .unwrap_or(haystack.len())
}

#[inline(always)]
fn contains(bits: &[u64; 4], byte: u8) -> bool {
    // Kernel entry points receive the membership bits: the high two bits select
    // a u64 entry and the low six bits select the byte's membership bit.
    bits[(byte >> 6) as usize] & (1 << (byte & 0b0011_1111)) != 0
}

// Every classifier consumes WIDTH valid bytes and returns one bit per byte.
// Entry points handle shorter inputs before entering these loops. Keeping
// directions separate makes the order of overlapping blocks explicit.
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
unsafe fn scan<const WIDTH: usize>(haystack: &[u8], classify: impl Fn(*const u8) -> u64) -> usize {
    debug_assert!(haystack.len() >= WIDTH);
    let base = haystack.as_ptr();
    let last = haystack.len() - WIDTH;
    let mut start = 0;
    loop {
        // SAFETY: start <= last, so the classifier can read WIDTH bytes at
        // this address without crossing either end of the input slice.
        let mask = classify(unsafe { base.add(start) });
        if mask != 0 {
            return start + mask.trailing_zeros() as usize;
        }
        if start == last {
            return haystack.len();
        }
        // The last load may overlap an already rejected block. Those bytes
        // cannot produce a match, so trailing_zeros still finds the first hit.
        start = (start + WIDTH).min(last);
    }
}

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
unsafe fn rscan<const WIDTH: usize>(haystack: &[u8], classify: impl Fn(*const u8) -> u64) -> usize {
    debug_assert!(haystack.len() >= WIDTH);
    let base = haystack.as_ptr();
    let mut start = haystack.len() - WIDTH;
    loop {
        // SAFETY: start never increases and is at most len - WIDTH. Each
        // load therefore stays entirely inside the input slice.
        let mask = classify(unsafe { base.add(start) });
        if mask != 0 {
            return start + (63 - mask.leading_zeros()) as usize;
        }
        if start == 0 {
            return haystack.len();
        }
        // The prefix may overlap a rejected block. Its highest selected bit
        // is still the last match because higher positions were all rejected.
        start = start.saturating_sub(WIDTH);
    }
}
