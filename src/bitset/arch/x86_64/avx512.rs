#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::super::{rscan, scan};
use super::avx2::{bitset_avx2, rbitset_avx2};

#[target_feature(enable = "avx512bw")]
pub(in super::super) fn bitset_avx512<const REVERT: bool>(
    bits: &[u64; 4],
    haystack: &[u8],
) -> usize {
    // SAFETY: the caller checked this backend. Narrower kernels require only
    // features implied by this entry point. scan/rscan supply in-bounds blocks.
    unsafe {
        if haystack.len() < 64 {
            return bitset_avx2::<REVERT>(bits, haystack);
        }
        // AVX-512BW shuffles within 128-bit lanes. Repeat both halves in all
        // four lanes; an input byte's top bit later selects the required half.
        let lower = _mm512_broadcast_i32x4(_mm_loadu_si128(bits.as_ptr().cast()));
        let upper = _mm512_broadcast_i32x4(_mm_loadu_si128(bits.as_ptr().add(2).cast()));
        let bit_masks = _mm512_broadcast_i32x4(_mm_setr_epi8(
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
        ));
        scan::<64>(haystack, |ptr| {
            // The scanner guarantees a complete 64-byte block; no alignment
            // is required by this load. Every lane retains input byte order.
            let bytes = _mm512_loadu_si512(ptr.cast());
            mask64::<REVERT>(lower, upper, bit_masks, bytes)
        })
    }
}

#[target_feature(enable = "avx512bw")]
pub(in super::super) fn rbitset_avx512<const REVERT: bool>(
    bits: &[u64; 4],
    haystack: &[u8],
) -> usize {
    // SAFETY: the caller checked this backend. Narrower kernels require only
    // features implied by this entry point. scan/rscan supply in-bounds blocks.
    unsafe {
        if haystack.len() < 64 {
            return rbitset_avx2::<REVERT>(bits, haystack);
        }
        // AVX-512BW shuffles within 128-bit lanes. Repeat both halves in all
        // four lanes; an input byte's top bit later selects the required half.
        let lower = _mm512_broadcast_i32x4(_mm_loadu_si128(bits.as_ptr().cast()));
        let upper = _mm512_broadcast_i32x4(_mm_loadu_si128(bits.as_ptr().add(2).cast()));
        let bit_masks = _mm512_broadcast_i32x4(_mm_setr_epi8(
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
        ));
        rscan::<64>(haystack, |ptr| {
            // The scanner guarantees a complete 64-byte block; no alignment
            // is required by this load. Every lane retains input byte order.
            let bytes = _mm512_loadu_si512(ptr.cast());
            mask64::<REVERT>(lower, upper, bit_masks, bytes)
        })
    }
}

#[inline]
#[target_feature(enable = "avx512bw")]
fn mask64<const REVERT: bool>(
    lower: __m512i,
    upper: __m512i,
    bit_masks: __m512i,
    bytes: __m512i,
) -> u64 {
    // The mask removes adjacent-byte spill after each 16-bit lane shifts.
    let indices = _mm512_and_si512(_mm512_srli_epi16::<3>(bytes), _mm512_set1_epi8(0b00011111));
    // MOVEMASK extracts b's top bit; masked blend selects the upper lookup
    // exactly where b >= 128. Each PSHUFB indexes its own 16-byte lane.
    let entry = _mm512_mask_blend_epi8(
        _mm512_movepi8_mask(bytes),
        _mm512_shuffle_epi8(lower, indices),
        _mm512_shuffle_epi8(upper, indices),
    );
    // Lookup b's low-three-bit choice, then compare the tested bit to zero.
    // AVX-512 returns comparison results directly as a 64-bit lane mask.
    let bit = _mm512_shuffle_epi8(
        bit_masks,
        _mm512_and_si512(bytes, _mm512_set1_epi8(0b00000111)),
    );
    let flags = _mm512_and_si512(entry, bit);
    let zero = _mm512_setzero_si512();
    // Each comparison sets scalar bit i exactly when byte lane i is selected.
    if REVERT {
        _mm512_cmpeq_epi8_mask(flags, zero) // mismatch: select nonmembers.
    } else {
        _mm512_cmpneq_epi8_mask(flags, zero) // find: select members.
    }
}

#[cfg(test)]
mod tests {
    use super::{bitset_avx512, rbitset_avx512};
    use crate::{bitset::BitsetTable, SimdBackend};

    #[test]
    fn bitset_avx512_finds_the_first_member() {
        if !SimdBackend::Avx512.is_available() {
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
            let actual = unsafe { bitset_avx512::<false>(&bits, &haystack) };
            assert_eq!(actual, 17, "member={member}");
        }
    }

    #[test]
    fn bitset_avx512_finds_the_first_nonmember() {
        if !SimdBackend::Avx512.is_available() {
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
            let actual = unsafe { bitset_avx512::<true>(&bits, &haystack) };
            assert_eq!(actual, 17, "member={member}");
        }
    }

    #[test]
    fn rbitset_avx512_finds_the_last_member() {
        if !SimdBackend::Avx512.is_available() {
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
            let actual = unsafe { rbitset_avx512::<false>(&bits, &haystack) };
            assert_eq!(actual, 96, "member={member}");
        }
    }

    #[test]
    fn rbitset_avx512_finds_the_last_nonmember() {
        if !SimdBackend::Avx512.is_available() {
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
            let actual = unsafe { rbitset_avx512::<true>(&bits, &haystack) };
            assert_eq!(actual, 96, "member={member}");
        }
    }
}
