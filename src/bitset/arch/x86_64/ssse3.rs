#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::super::{rscan, rsearch_fallback, scan, search_fallback};

#[target_feature(enable = "ssse3")]
pub(in super::super) unsafe fn bitset_ssse3<const REVERT: bool>(
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
            1 << 0,
            1 << 1,
            1 << 2,
            1 << 3,
            1 << 4,
            1 << 5,
            1 << 6,
            (1u8 << 7) as i8,
            1 << 0,
            1 << 1,
            1 << 2,
            1 << 3,
            1 << 4,
            1 << 5,
            1 << 6,
            (1u8 << 7) as i8,
        );
        scan::<16>(haystack, |ptr| {
            // The scanner guarantees a complete 16-byte block; no alignment
            // is required by this load. Every lane retains input byte order.
            let bytes = _mm_loadu_si128(ptr.cast());
            mask16::<REVERT>(lower, upper, bit_masks, bytes)
        })
    }
}

#[target_feature(enable = "ssse3")]
pub(in super::super) unsafe fn rbitset_ssse3<const REVERT: bool>(
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
            1 << 0,
            1 << 1,
            1 << 2,
            1 << 3,
            1 << 4,
            1 << 5,
            1 << 6,
            (1u8 << 7) as i8,
            1 << 0,
            1 << 1,
            1 << 2,
            1 << 3,
            1 << 4,
            1 << 5,
            1 << 6,
            (1u8 << 7) as i8,
        );
        rscan::<16>(haystack, |ptr| {
            // The scanner guarantees a complete 16-byte block; no alignment
            // is required by this load. Every lane retains input byte order.
            let bytes = _mm_loadu_si128(ptr.cast());
            mask16::<REVERT>(lower, upper, bit_masks, bytes)
        })
    }
}

#[inline]
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn mask16<const REVERT: bool>(
    lower: __m128i,
    upper: __m128i,
    bit_masks: __m128i,
    bytes: __m128i,
) -> u64 {
    // Shift 16-bit lanes, then clear the three bits that may spill from the
    // neighboring byte. Every resulting byte is exactly b >> 3, in 0..=31.
    let indices = _mm_and_si128(_mm_srli_epi16::<3>(bytes), _mm_set1_epi8(0b00011111));
    // PSHUFB uses the low four index bits. Lookup both halves; a signed-negative
    // byte (b >= 128) selects the upper half, covering all 32 bitset bytes.
    let lower = _mm_shuffle_epi8(lower, indices);
    let upper = _mm_shuffle_epi8(upper, indices);
    let high_half = _mm_cmpgt_epi8(_mm_setzero_si128(), bytes);
    let entry = _mm_or_si128(
        _mm_and_si128(high_half, upper),
        _mm_andnot_si128(high_half, lower),
    );
    // b & 7 selects 1 << (b & 7). AND tests that one membership bit;
    // comparison with zero produces 0b11111111 for each nonmember.
    let bit = _mm_shuffle_epi8(bit_masks, _mm_and_si128(bytes, _mm_set1_epi8(0b00000111)));
    let absent = _mm_cmpeq_epi8(_mm_and_si128(entry, bit), _mm_setzero_si128());
    // Pack each byte's top bit into bit i of a u16. Invert only those 16 bits
    // for membership; widening afterwards cannot create nonexistent lanes.
    let absent = _mm_movemask_epi8(absent) as u16;
    if REVERT {
        absent as u64
    } else {
        !absent as u64
    }
}

#[cfg(test)]
mod tests {
    use super::{bitset_ssse3, rbitset_ssse3};
    use crate::{bitset::BitsetTable, SimdBackend};

    #[test]
    fn bitset_ssse3_finds_the_first_member() {
        if !SimdBackend::Ssse3.is_available() {
            return;
        }
        // Each pair is the first and last byte represented by one u64 entry.
        let needle = [
            0, 63, // bits[0]: 0..=63
            64, 127, // bits[1]: 64..=127
            128, 191, // bits[2]: 128..=191
            192, 255, // bits[3]: 192..=255
        ];
        let bits = *BitsetTable::new(&needle).bits();
        for member in needle {
            // Flipping bit 0 of any member above produces a nonmember.
            let mut haystack = [member ^ 1; 128];
            haystack[17] = member;
            haystack[96] = member;
            // SAFETY: the availability check covers this kernel's instructions.
            let actual = unsafe { bitset_ssse3::<false>(&bits, &haystack) };
            assert_eq!(actual, 17, "member={member}");
        }
    }

    #[test]
    fn bitset_ssse3_finds_the_first_nonmember() {
        if !SimdBackend::Ssse3.is_available() {
            return;
        }
        // Each pair is the first and last byte represented by one u64 entry.
        let needle = [
            0, 63, // bits[0]: 0..=63
            64, 127, // bits[1]: 64..=127
            128, 191, // bits[2]: 128..=191
            192, 255, // bits[3]: 192..=255
        ];
        let bits = *BitsetTable::new(&needle).bits();
        for member in needle {
            // Flipping bit 0 of any member above produces a nonmember.
            let mut haystack = [member; 128];
            haystack[17] = member ^ 1;
            haystack[96] = member ^ 1;
            // SAFETY: the availability check covers this kernel's instructions.
            let actual = unsafe { bitset_ssse3::<true>(&bits, &haystack) };
            assert_eq!(actual, 17, "member={member}");
        }
    }

    #[test]
    fn rbitset_ssse3_finds_the_last_member() {
        if !SimdBackend::Ssse3.is_available() {
            return;
        }
        // Each pair is the first and last byte represented by one u64 entry.
        let needle = [
            0, 63, // bits[0]: 0..=63
            64, 127, // bits[1]: 64..=127
            128, 191, // bits[2]: 128..=191
            192, 255, // bits[3]: 192..=255
        ];
        let bits = *BitsetTable::new(&needle).bits();
        for member in needle {
            // Flipping bit 0 of any member above produces a nonmember.
            let mut haystack = [member ^ 1; 128];
            haystack[17] = member;
            haystack[96] = member;
            // SAFETY: the availability check covers this kernel's instructions.
            let actual = unsafe { rbitset_ssse3::<false>(&bits, &haystack) };
            assert_eq!(actual, 96, "member={member}");
        }
    }

    #[test]
    fn rbitset_ssse3_finds_the_last_nonmember() {
        if !SimdBackend::Ssse3.is_available() {
            return;
        }
        // Each pair is the first and last byte represented by one u64 entry.
        let needle = [
            0, 63, // bits[0]: 0..=63
            64, 127, // bits[1]: 64..=127
            128, 191, // bits[2]: 128..=191
            192, 255, // bits[3]: 192..=255
        ];
        let bits = *BitsetTable::new(&needle).bits();
        for member in needle {
            // Flipping bit 0 of any member above produces a nonmember.
            let mut haystack = [member; 128];
            haystack[17] = member ^ 1;
            haystack[96] = member ^ 1;
            // SAFETY: the availability check covers this kernel's instructions.
            let actual = unsafe { rbitset_ssse3::<true>(&bits, &haystack) };
            assert_eq!(actual, 96, "member={member}");
        }
    }
}
