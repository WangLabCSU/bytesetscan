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
// Function pointers do not carry target_feature requirements, so calling one
// remains unsafe even when its target is declared as a safe function.
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
// Entry points use WIDTH = 16, 32, or 64 and handle shorter inputs first.
// Boundary loads may be unaligned; interior loads use WIDTH-byte alignment.
// Each direction checks its boundary block before rounding the address, so
// any overlap contains only bytes already known not to match.
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
unsafe fn scan<const WIDTH: usize>(haystack: &[u8], classify: impl Fn(*const u8) -> u64) -> usize {
    debug_assert!(haystack.len() >= WIDTH);
    let ptr = haystack.as_ptr();
    let len = haystack.len();
    let last = len - WIDTH;
    // The first block may be unaligned. The entry point guarantees WIDTH bytes.
    let mask = classify(ptr);
    if mask != 0 {
        return mask.trailing_zeros() as usize;
    }
    if last == 0 {
        return len;
    }
    // WIDTH is a power of two, so masking with WIDTH - 1 gives the address's
    // remainder modulo WIDTH. Subtract it from WIDTH to reach the next boundary;
    // an already aligned ptr advances by WIDTH because its block was checked.
    let mut start = WIDTH - (ptr as usize & (WIDTH - 1));
    while start < last {
        // SAFETY: start < len - WIDTH, so the full aligned block lies inside
        // haystack. Advancing by WIDTH preserves alignment without skipping bytes.
        let mask = classify(unsafe { ptr.add(start) });
        if mask != 0 {
            return start + mask.trailing_zeros() as usize;
        }
        start += WIDTH;
    }
    // SAFETY: last = len - WIDTH makes this full, possibly unaligned suffix
    // end exactly at the slice boundary. Any overlap was already rejected.
    let mask = classify(unsafe { ptr.add(last) });
    if mask != 0 {
        last + mask.trailing_zeros() as usize
    } else {
        len
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
    let ptr = haystack.as_ptr();
    let len = haystack.len();
    let last = len - WIDTH;
    // SAFETY: last = len - WIDTH places this full block at the end of haystack.
    let mask = classify(unsafe { ptr.add(last) });
    if mask != 0 {
        return last + (63 - mask.leading_zeros()) as usize;
    }
    if last == 0 {
        return len;
    }
    // The end address's remainder modulo WIDTH gives its distance past the
    // preceding aligned boundary. Subtract it from len to align ptr + end.
    // SAFETY: adding len gives the valid one-past-the-end pointer.
    let remainder = unsafe { ptr.add(len) } as usize & (WIDTH - 1);
    let mut end = len - remainder;
    if remainder == 0 {
        // The suffix was already aligned and checked. Move past that block
        // before the loop decrements end to the next block's start.
        end -= WIDTH;
    }
    while end > WIDTH {
        end -= WIDTH;
        // SAFETY: this aligned block precedes the checked suffix and remains
        // inside haystack. Decreasing end by WIDTH preserves alignment.
        let mask = classify(unsafe { ptr.add(end) });
        if mask != 0 {
            return end + (63 - mask.leading_zeros()) as usize;
        }
    }
    // The prefix may be unaligned and overlap a rejected block. All WIDTH bytes
    // are inside haystack, and leading_zeros still identifies the last match.
    let mask = classify(ptr);
    if mask != 0 {
        (63 - mask.leading_zeros()) as usize
    } else {
        len
    }
}

#[cfg(all(
    test,
    any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )
))]
mod tests {
    extern crate std;

    use std::{cell::RefCell, vec::Vec};

    use super::{rscan, scan};

    // Explicit alignment makes every expected block offset independent of the
    // stack address. The scans may still receive a deliberately unaligned slice.
    #[repr(align(64))]
    struct Aligned([u8; 256]);

    #[test]
    fn scan_aligns_interior_blocks() {
        let storage = Aligned([0; 256]);
        let haystack = &storage.0[3..200];
        let ptr = haystack.as_ptr();
        let visited = RefCell::new(Vec::new());
        // SAFETY: haystack contains at least 64 bytes. Returning zero makes
        // scan visit every block without requiring a particular SIMD backend.
        let end = unsafe {
            scan::<64>(haystack, |block_ptr| {
                visited.borrow_mut().push(block_ptr as usize - ptr as usize);
                0
            })
        };
        // The two interior loads reach storage offsets 64 and 128. Only the
        // first and last loads keep the slice's unaligned boundaries.
        assert_eq!(visited.into_inner(), [0, 61, 125, 133]);
        assert_eq!(end, haystack.len());
    }

    #[test]
    fn scan_does_not_repeat_an_aligned_prefix() {
        let storage = Aligned([0; 256]);
        let haystack = &storage.0[..192];
        let ptr = haystack.as_ptr();
        let visited = RefCell::new(Vec::new());
        // SAFETY: the slice contains three complete 64-byte blocks.
        unsafe {
            scan::<64>(haystack, |block_ptr| {
                visited.borrow_mut().push(block_ptr as usize - ptr as usize);
                0
            });
        }
        assert_eq!(visited.into_inner(), [0, 64, 128]);
    }

    #[test]
    fn scan_reads_a_single_block_once() {
        let storage = Aligned([0; 256]);
        let haystack = &storage.0[3..67];
        let ptr = haystack.as_ptr();
        let visited = RefCell::new(Vec::new());
        // SAFETY: the unaligned slice contains exactly one complete block.
        unsafe {
            scan::<64>(haystack, |block_ptr| {
                visited.borrow_mut().push(block_ptr as usize - ptr as usize);
                0
            });
        }
        assert_eq!(visited.into_inner(), [0]);
    }

    #[test]
    fn rscan_aligns_interior_blocks() {
        let storage = Aligned([0; 256]);
        let haystack = &storage.0[3..200];
        let ptr = haystack.as_ptr();
        let visited = RefCell::new(Vec::new());
        // SAFETY: haystack contains at least 64 bytes. Zero masks keep the
        // reverse scan running through its aligned interior and final prefix.
        let end = unsafe {
            rscan::<64>(haystack, |block_ptr| {
                visited.borrow_mut().push(block_ptr as usize - ptr as usize);
                0
            })
        };
        assert_eq!(visited.into_inner(), [133, 125, 61, 0]);
        assert_eq!(end, haystack.len());
    }

    #[test]
    fn rscan_does_not_repeat_an_aligned_suffix() {
        let storage = Aligned([0; 256]);
        let haystack = &storage.0[..192];
        let ptr = haystack.as_ptr();
        let visited = RefCell::new(Vec::new());
        // SAFETY: the slice contains three complete 64-byte blocks.
        unsafe {
            rscan::<64>(haystack, |block_ptr| {
                visited.borrow_mut().push(block_ptr as usize - ptr as usize);
                0
            });
        }
        assert_eq!(visited.into_inner(), [128, 64, 0]);
    }

    #[test]
    fn rscan_reads_a_single_block_once() {
        let storage = Aligned([0; 256]);
        let haystack = &storage.0[3..67];
        let ptr = haystack.as_ptr();
        let visited = RefCell::new(Vec::new());
        // SAFETY: the unaligned slice contains exactly one complete block.
        unsafe {
            rscan::<64>(haystack, |block_ptr| {
                visited.borrow_mut().push(block_ptr as usize - ptr as usize);
                0
            });
        }
        assert_eq!(visited.into_inner(), [0]);
    }
}
