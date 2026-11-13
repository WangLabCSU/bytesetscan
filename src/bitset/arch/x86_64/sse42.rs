#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::super::{rscan, rsearch_fallback, scan, search_fallback};
use super::ssse3::mask16;

#[target_feature(enable = "ssse3,sse4.2,popcnt")]
pub(in super::super) unsafe fn bitset_sse42<const REVERT: bool>(
    bits: &[u64; 4],
    haystack: &[u8],
) -> usize {
    // SAFETY: the caller checked this backend. Narrower kernels require only
    // features implied by this entry point. scan/rscan supply in-bounds blocks.
    unsafe {
        if haystack.len() < 16 {
            return search_fallback::<REVERT>(bits, haystack);
        }
        // Two unaligned loads read the bitset bytes for 0..127 and 128..255.
        let lower = _mm_loadu_si128(bits.as_ptr().cast());
        let upper = _mm_loadu_si128(bits.as_ptr().add(2).cast());
        // Each index 0..7 maps to the single bit to test in the chosen byte.
        let bit_masks = _mm_setr_epi8(
            0b00000001,
            0b00000010,
            0b00000100,
            0b00001000,
            0b00010000,
            0b00100000,
            0b01000000,
            0b10000000u8 as i8,
            0b00000001,
            0b00000010,
            0b00000100,
            0b00001000,
            0b00010000,
            0b00100000,
            0b01000000,
            0b10000000u8 as i8,
        );
        scan::<16>(haystack, |ptr| {
            // The scanner guarantees a complete 16-byte block; no alignment
            // is required by this load. Every lane retains input byte order.
            let bytes = _mm_loadu_si128(ptr.cast());
            mask16::<REVERT>(lower, upper, bit_masks, bytes)
        })
    }
}

#[target_feature(enable = "ssse3,sse4.2,popcnt")]
pub(in super::super) unsafe fn rbitset_sse42<const REVERT: bool>(
    bits: &[u64; 4],
    haystack: &[u8],
) -> usize {
    // SAFETY: the caller checked this backend. Narrower kernels require only
    // features implied by this entry point. scan/rscan supply in-bounds blocks.
    unsafe {
        if haystack.len() < 16 {
            return rsearch_fallback::<REVERT>(bits, haystack);
        }
        // Two unaligned loads read the bitset bytes for 0..127 and 128..255.
        let lower = _mm_loadu_si128(bits.as_ptr().cast());
        let upper = _mm_loadu_si128(bits.as_ptr().add(2).cast());
        // Each index 0..7 maps to the single bit to test in the chosen byte.
        let bit_masks = _mm_setr_epi8(
            0b00000001,
            0b00000010,
            0b00000100,
            0b00001000,
            0b00010000,
            0b00100000,
            0b01000000,
            0b10000000u8 as i8,
            0b00000001,
            0b00000010,
            0b00000100,
            0b00001000,
            0b00010000,
            0b00100000,
            0b01000000,
            0b10000000u8 as i8,
        );
        rscan::<16>(haystack, |ptr| {
            // The scanner guarantees a complete 16-byte block; no alignment
            // is required by this load. Every lane retains input byte order.
            let bytes = _mm_loadu_si128(ptr.cast());
            mask16::<REVERT>(lower, upper, bit_masks, bytes)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{bitset_sse42, rbitset_sse42};
    use crate::{bitset::BitsetTable, SimdBackend};

    #[test]
    fn bitset_sse42_finds_the_first_member() {
        if !SimdBackend::Sse4_2.is_available() {
            return;
        }
        // Members sit on both sides of each 64-bit word boundary.
        let needle = [0, 63, 64, 127, 128, 191, 192, 255];
        let bits = *BitsetTable::new(&needle).bits();
        for member in needle {
            // Flipping bit 0 of any member above produces a nonmember.
            let mut haystack = [member ^ 1; 128];
            haystack[17] = member;
            haystack[96] = member;
            // SAFETY: the availability check covers this kernel's instructions.
            let actual = unsafe { bitset_sse42::<false>(&bits, &haystack) };
            assert_eq!(actual, 17, "member={member}");
        }
    }

    #[test]
    fn bitset_sse42_finds_the_first_nonmember() {
        if !SimdBackend::Sse4_2.is_available() {
            return;
        }
        // Members sit on both sides of each 64-bit word boundary.
        let needle = [0, 63, 64, 127, 128, 191, 192, 255];
        let bits = *BitsetTable::new(&needle).bits();
        for member in needle {
            // Flipping bit 0 of any member above produces a nonmember.
            let mut haystack = [member; 128];
            haystack[17] = member ^ 1;
            haystack[96] = member ^ 1;
            // SAFETY: the availability check covers this kernel's instructions.
            let actual = unsafe { bitset_sse42::<true>(&bits, &haystack) };
            assert_eq!(actual, 17, "member={member}");
        }
    }

    #[test]
    fn rbitset_sse42_finds_the_last_member() {
        if !SimdBackend::Sse4_2.is_available() {
            return;
        }
        // Members sit on both sides of each 64-bit word boundary.
        let needle = [0, 63, 64, 127, 128, 191, 192, 255];
        let bits = *BitsetTable::new(&needle).bits();
        for member in needle {
            // Flipping bit 0 of any member above produces a nonmember.
            let mut haystack = [member ^ 1; 128];
            haystack[17] = member;
            haystack[96] = member;
            // SAFETY: the availability check covers this kernel's instructions.
            let actual = unsafe { rbitset_sse42::<false>(&bits, &haystack) };
            assert_eq!(actual, 96, "member={member}");
        }
    }

    #[test]
    fn rbitset_sse42_finds_the_last_nonmember() {
        if !SimdBackend::Sse4_2.is_available() {
            return;
        }
        // Members sit on both sides of each 64-bit word boundary.
        let needle = [0, 63, 64, 127, 128, 191, 192, 255];
        let bits = *BitsetTable::new(&needle).bits();
        for member in needle {
            // Flipping bit 0 of any member above produces a nonmember.
            let mut haystack = [member; 128];
            haystack[17] = member ^ 1;
            haystack[96] = member ^ 1;
            // SAFETY: the availability check covers this kernel's instructions.
            let actual = unsafe { rbitset_sse42::<true>(&bits, &haystack) };
            assert_eq!(actual, 96, "member={member}");
        }
    }
}
