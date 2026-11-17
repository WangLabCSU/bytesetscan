// Automatic and explicitly configured searches share kernel selection.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_64;
#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm32;

use crate::backend::SimdBackend;

// Each function returns None when no match is found. Before calling
// a SIMD kernel, establish backend availability; entry points handle all lengths.
// Selecting or storing a pointer does not execute its target instructions.
// Function pointers do not carry target_feature requirements, so calling one
// remains unsafe even when its target is declared as a safe function.
pub(super) type SearchFn = unsafe fn(&[u8; 16], &[u8; 16], &[u8]) -> Option<usize>;

// Used by automatic dispatch and ShuftiEngine construction. Keeping selection
// separate from execution lets an engine retain the kernel without scanning
// any input. Callers must validate the backend before invoking the result.
#[inline(always)]
pub(super) fn search_fn<const REVERT: bool>(backend: SimdBackend) -> SearchFn {
    match backend {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Avx2 => x86_64::shufti_avx2::<REVERT>,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Ssse3 | SimdBackend::Sse4_2 => x86_64::shufti_ssse3::<REVERT>,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Avx512 | SimdBackend::Avx512Vbmi => x86_64::shufti_avx512::<REVERT>,
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => aarch64::search::<REVERT>,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        SimdBackend::WasmSimd128 => wasm32::search::<REVERT>,
        _ => search_fallback::<REVERT>,
    }
}

#[inline(always)]
pub(super) fn search_fallback<const REVERT: bool>(
    low: &[u8; 16],
    high: &[u8; 16],
    haystack: &[u8],
) -> Option<usize> {
    // Check membership one byte at a time and stop at the first selected byte.
    // An empty or exhausted slice returns None.
    haystack
        .iter()
        .position(|&byte| find::<REVERT>(low, high, byte))
}

// Reverse scans have their own entry points; membership and nonmembership
// remain the only const-generic choice. As with search_fn, selection alone
// does not establish that the returned SIMD function can be called safely.
#[inline(always)]
pub(super) fn rsearch_fn<const REVERT: bool>(backend: SimdBackend) -> SearchFn {
    match backend {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Avx2 => x86_64::rshufti_avx2::<REVERT>,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Ssse3 | SimdBackend::Sse4_2 => x86_64::rshufti_ssse3::<REVERT>,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Avx512 | SimdBackend::Avx512Vbmi => x86_64::rshufti_avx512::<REVERT>,
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => aarch64::rsearch::<REVERT>,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        SimdBackend::WasmSimd128 => wasm32::rsearch::<REVERT>,
        _ => rsearch_fallback::<REVERT>,
    }
}

#[inline(always)]
pub(super) fn rsearch_fallback<const REVERT: bool>(
    low: &[u8; 16],
    high: &[u8; 16],
    haystack: &[u8],
) -> Option<usize> {
    haystack
        .iter()
        .rposition(|&byte| find::<REVERT>(low, high, byte))
}

#[inline(always)]
fn find<const REVERT: bool>(low: &[u8; 16], high: &[u8; 16], byte: u8) -> bool {
    // low lists the groups allowing this low value; high selects the
    // group for this high value. A shared flag means the byte is allowed.
    // For a table built from b"ABRabr", B uses low[2] & high[4] =
    // 0b00000011 & 0b00000001;
    // R uses low[2] & high[5] = 0b00000011 & 0b00000010, accepting it;
    // Q uses low[1] & high[5] = 0b00000001 & 0b00000010 = 0b00000000.
    // SAFETY: masking a byte to its low four bits produces an index in 0..16.
    let lo = unsafe { low.get_unchecked((byte & 0b0000_1111) as usize) };
    // SAFETY: shifting a u8 right by four also produces an index in 0..16.
    let hi = unsafe { high.get_unchecked((byte >> 4) as usize) };

    (lo & hi != 0) != REVERT
}
