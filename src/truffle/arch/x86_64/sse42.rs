#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::ssse3::{rsearch128, search128};

// Requires ssse3,sse4.2,popcnt, valid table arrays, any slice length.
#[target_feature(enable = "ssse3,sse4.2,popcnt")]
pub(in super::super) fn truffle_sse42<const REVERT: bool>(
    lower: &[u8; 16],
    upper: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // SAFETY: dispatch establishes CPU support; the scan handles short inputs.
    // Both table arrays provide one initialized 16-byte vector.
    unsafe {
        search128::<REVERT>(
            // Load the 16 initialized table entries into one register; the
            // array has byte alignment, so use the unaligned load intrinsic.
            _mm_loadu_si128(lower.as_ptr().cast()),
            // Load the 16 initialized table entries into one register; the
            // array has byte alignment, so use the unaligned load intrinsic.
            _mm_loadu_si128(upper.as_ptr().cast()),
            haystack,
        )
    }
}

// Reverse entry point. Requires ssse3,sse4.2,popcnt any slice length.
#[target_feature(enable = "ssse3,sse4.2,popcnt")]
pub(in super::super) fn rtruffle_sse42<const REVERT: bool>(
    lower: &[u8; 16],
    upper: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // SAFETY: dispatch establishes CPU support; the scan handles short inputs.
    // Both table arrays provide one initialized 16-byte vector.
    unsafe {
        rsearch128::<REVERT>(
            // Load the 16 initialized table entries into one register; the
            // array has byte alignment, so use the unaligned load intrinsic.
            _mm_loadu_si128(lower.as_ptr().cast()),
            // Load the 16 initialized table entries into one register; the
            // array has byte alignment, so use the unaligned load intrinsic.
            _mm_loadu_si128(upper.as_ptr().cast()),
            haystack,
        )
    }
}
