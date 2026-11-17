// Automatic and explicitly configured searches share kernel selection.

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm32;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_64;

use crate::backend::SimdBackend;

// Each function returns None when no match is found. Before calling
// a SIMD kernel, establish backend availability; entry points handle all lengths.
// Selecting or storing a pointer does not execute its target instructions.
// Function pointers do not carry target_feature requirements, so calling one
// remains unsafe even when its target is declared as a safe function.
pub(super) type SearchFn = unsafe fn(&[u8; 16], &[u8; 16], &[u8]) -> Option<usize>;

// Used by automatic dispatch and TruffleEngine construction. Keeping selection
// separate from execution lets an engine retain the kernel without scanning
// any input. Callers must validate the backend before invoking the result.
#[inline(always)]
pub(super) fn search_fn<const REVERT: bool>(backend: SimdBackend) -> SearchFn {
    match backend {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Avx2 => x86_64::truffle_avx2::<REVERT>,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Ssse3 | SimdBackend::Sse4_2 => x86_64::truffle_ssse3::<REVERT>,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Avx512 | SimdBackend::Avx512Vbmi => x86_64::truffle_avx512::<REVERT>,
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => aarch64::search::<REVERT>,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        SimdBackend::WasmSimd128 => wasm32::search::<REVERT>,
        _ => search_fallback::<REVERT>,
    }
}

#[inline(always)]
pub(super) fn search_fallback<const REVERT: bool>(
    lower: &[u8; 16],
    upper: &[u8; 16],
    haystack: &[u8],
) -> Option<usize> {
    // An empty or exhausted slice returns None, as do the SIMD kernels.
    haystack
        .iter()
        .position(|&byte| find::<REVERT>(lower, upper, byte))
}

// Reverse scans have their own entry points; membership and nonmembership
// remain the only const-generic choice. As with search_fn, selection alone
// does not establish that the returned SIMD function can be called safely.
#[inline(always)]
pub(super) fn rsearch_fn<const REVERT: bool>(backend: SimdBackend) -> SearchFn {
    match backend {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Avx2 => x86_64::rtruffle_avx2::<REVERT>,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Ssse3 | SimdBackend::Sse4_2 => x86_64::rtruffle_ssse3::<REVERT>,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Avx512 | SimdBackend::Avx512Vbmi => x86_64::rtruffle_avx512::<REVERT>,
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => aarch64::rsearch::<REVERT>,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        SimdBackend::WasmSimd128 => wasm32::rsearch::<REVERT>,
        _ => rsearch_fallback::<REVERT>,
    }
}

#[inline(always)]
pub(super) fn rsearch_fallback<const REVERT: bool>(
    lower: &[u8; 16],
    upper: &[u8; 16],
    haystack: &[u8],
) -> Option<usize> {
    haystack
        .iter()
        .rposition(|&byte| find::<REVERT>(lower, upper, byte))
}

#[inline(always)]
fn find<const REVERT: bool>(lower: &[u8; 16], upper: &[u8; 16], byte: u8) -> bool {
    // The top bit chooses the table half. Bits 0..3 index its byte and
    // bits 4..6 select the membership bit, covering all 256 possible values.
    let half = if byte < 128 { lower } else { upper };
    //SAFETY: we only extract low nibble as the index, it must < 16
    let flags = unsafe { half.get_unchecked((byte & 0b0000_1111) as usize) };
    (flags & (1 << ((byte >> 4) & 0b0000_0111)) != 0) != REVERT
}
