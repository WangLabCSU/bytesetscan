#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::super::{rscan, scan};
use super::ssse3::{bitset_ssse3, rbitset_ssse3};

#[target_feature(enable = "avx2")]
pub(in super::super) unsafe fn bitset_avx2<const REVERT: bool>(
    bits: &[u64; 4],
    haystack: &[u8],
) -> usize {
    // SAFETY: the caller checked this backend. Narrower kernels require only
    // features implied by this entry point. scan/rscan supply in-bounds blocks.
    unsafe {
        if haystack.len() < 32 {
            return bitset_ssse3::<REVERT>(bits, haystack);
        }
        // Each 128-bit lane has its own byte shuffle. Repeat both table halves
        // and the eight bit choices so every lane can classify independently.
        let lower = _mm256_broadcastsi128_si256(_mm_loadu_si128(bits.as_ptr().cast()));
        let upper = _mm256_broadcastsi128_si256(_mm_loadu_si128(bits.as_ptr().add(2).cast()));
        let bit_masks = _mm256_broadcastsi128_si256(_mm_setr_epi8(
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
        ));
        scan::<32>(haystack, |ptr| {
            // The scanner guarantees a complete 32-byte block; no alignment
            // is required by this load. Every lane retains input byte order.
            let bytes = _mm256_loadu_si256(ptr.cast());
            mask32::<REVERT>(lower, upper, bit_masks, bytes)
        })
    }
}

#[target_feature(enable = "avx2")]
pub(in super::super) unsafe fn rbitset_avx2<const REVERT: bool>(
    bits: &[u64; 4],
    haystack: &[u8],
) -> usize {
    // SAFETY: the caller checked this backend. Narrower kernels require only
    // features implied by this entry point. scan/rscan supply in-bounds blocks.
    unsafe {
        if haystack.len() < 32 {
            return rbitset_ssse3::<REVERT>(bits, haystack);
        }
        // Each 128-bit lane has its own byte shuffle. Repeat both table halves
        // and the eight bit choices so every lane can classify independently.
        let lower = _mm256_broadcastsi128_si256(_mm_loadu_si128(bits.as_ptr().cast()));
        let upper = _mm256_broadcastsi128_si256(_mm_loadu_si128(bits.as_ptr().add(2).cast()));
        let bit_masks = _mm256_broadcastsi128_si256(_mm_setr_epi8(
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
        ));
        rscan::<32>(haystack, |ptr| {
            // The scanner guarantees a complete 32-byte block; no alignment
            // is required by this load. Every lane retains input byte order.
            let bytes = _mm256_loadu_si256(ptr.cast());
            mask32::<REVERT>(lower, upper, bit_masks, bytes)
        })
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn mask32<const REVERT: bool>(
    lower: __m256i,
    upper: __m256i,
    bit_masks: __m256i,
    bytes: __m256i,
) -> u64 {
    // Mask after the lane-wise shift removes neighboring-byte bits, giving
    // 32 independent indices b >> 3 into the bitset's memory bytes.
    let indices = _mm256_and_si256(_mm256_srli_epi16::<3>(bytes), _mm256_set1_epi8(0b00011111));
    // Both table halves were broadcast into each 128-bit shuffle lane.
    // BLENDV uses each input byte's sign bit, selecting upper for b >= 128.
    let entry = _mm256_blendv_epi8(
        _mm256_shuffle_epi8(lower, indices),
        _mm256_shuffle_epi8(upper, indices),
        bytes,
    );
    // Look up one membership bit per input byte, then classify zero as absent.
    let bit = _mm256_shuffle_epi8(
        bit_masks,
        _mm256_and_si256(bytes, _mm256_set1_epi8(0b00000111)),
    );
    let absent = _mm256_cmpeq_epi8(_mm256_and_si256(entry, bit), _mm256_setzero_si256());
    // MOVEMASK preserves lane order. Limit inversion to 32 real input lanes.
    let absent = _mm256_movemask_epi8(absent) as u32;
    if REVERT {
        absent as u64
    } else {
        !absent as u64
    }
}

#[cfg(test)]
mod tests {
    use super::{bitset_avx2, rbitset_avx2};
    use crate::{bitset::BitsetTable, SimdBackend};

    #[test]
    fn bitset_avx2_finds_the_first_member() {
        if !SimdBackend::Avx2.is_available() {
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
            let actual = unsafe { bitset_avx2::<false>(&bits, &haystack) };
            assert_eq!(actual, 17, "member={member}");
        }
    }

    #[test]
    fn bitset_avx2_finds_the_first_nonmember() {
        if !SimdBackend::Avx2.is_available() {
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
            let actual = unsafe { bitset_avx2::<true>(&bits, &haystack) };
            assert_eq!(actual, 17, "member={member}");
        }
    }

    #[test]
    fn rbitset_avx2_finds_the_last_member() {
        if !SimdBackend::Avx2.is_available() {
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
            let actual = unsafe { rbitset_avx2::<false>(&bits, &haystack) };
            assert_eq!(actual, 96, "member={member}");
        }
    }

    #[test]
    fn rbitset_avx2_finds_the_last_nonmember() {
        if !SimdBackend::Avx2.is_available() {
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
            let actual = unsafe { rbitset_avx2::<true>(&bits, &haystack) };
            assert_eq!(actual, 96, "member={member}");
        }
    }
}
