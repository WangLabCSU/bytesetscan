#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::super::{rscan, scan};
use super::avx2::{bitset_avx2, rbitset_avx2};

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vbmi")]
pub(in super::super) unsafe fn bitset_vbmi<const REVERT: bool>(
    bits: &[u64; 4],
    haystack: &[u8],
) -> usize {
    // SAFETY: the caller checked this backend. Narrower kernels require only
    // features implied by this entry point. scan/rscan supply in-bounds blocks.
    unsafe {
        if haystack.len() < 64 {
            return bitset_avx2::<REVERT>(bits, haystack);
        }
        // VBMI can index all 32 table bytes in one shuffle. Broadcasting the
        // four u64 entries fills 64 lanes; indices stay below 32, using the first copy.
        let table = _mm512_broadcast_i64x4(_mm256_loadu_si256(bits.as_ptr().cast()));
        // PSHUFB still works in 128-bit lanes, so repeat the bit choices four times.
        let bit_masks = _mm512_broadcast_i32x4(_mm_setr_epi8(
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
        scan::<64>(haystack, |ptr| {
            // The scanner guarantees a complete 64-byte block; no alignment
            // is required by this load. Every lane retains input byte order.
            let bytes = _mm512_loadu_si512(ptr.cast());
            mask64_vbmi::<REVERT>(table, bit_masks, bytes)
        })
    }
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vbmi")]
pub(in super::super) unsafe fn rbitset_vbmi<const REVERT: bool>(
    bits: &[u64; 4],
    haystack: &[u8],
) -> usize {
    // SAFETY: the caller checked this backend. Narrower kernels require only
    // features implied by this entry point. scan/rscan supply in-bounds blocks.
    unsafe {
        if haystack.len() < 64 {
            return rbitset_avx2::<REVERT>(bits, haystack);
        }
        // VBMI can index all 32 table bytes in one shuffle. Broadcasting the
        // four u64 entries fills 64 lanes; indices stay below 32, using the first copy.
        let table = _mm512_broadcast_i64x4(_mm256_loadu_si256(bits.as_ptr().cast()));
        // PSHUFB still works in 128-bit lanes, so repeat the bit choices four times.
        let bit_masks = _mm512_broadcast_i32x4(_mm_setr_epi8(
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
        rscan::<64>(haystack, |ptr| {
            // The scanner guarantees a complete 64-byte block; no alignment
            // is required by this load. Every lane retains input byte order.
            let bytes = _mm512_loadu_si512(ptr.cast());
            mask64_vbmi::<REVERT>(table, bit_masks, bytes)
        })
    }
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn mask64_vbmi<const REVERT: bool>(
    table: __m512i,
    bit_masks: __m512i,
    bytes: __m512i,
) -> u64 {
    // VPERMB indexes across the whole 64-byte register, so indices 0..31
    // directly choose bitset[b >> 3] without a separate table-half selection.
    let indices = _mm512_and_si512(_mm512_srli_epi16::<3>(bytes), _mm512_set1_epi8(0b00011111));
    let entry = _mm512_permutexvar_epi8(indices, table);
    // The low three bits select a membership bit in each retrieved table byte.
    let bit = _mm512_shuffle_epi8(
        bit_masks,
        _mm512_and_si512(bytes, _mm512_set1_epi8(0b00000111)),
    );
    let absent = _mm512_cmpeq_epi8_mask(_mm512_and_si512(entry, bit), _mm512_setzero_si512());
    // All 64 scalar bits correspond to input lanes; no padding bits exist.
    if REVERT {
        absent
    } else {
        !absent
    }
}

#[cfg(test)]
mod tests {
    use super::{bitset_vbmi, rbitset_vbmi};
    use crate::{bitset::BitsetTable, SimdBackend};

    #[test]
    fn bitset_vbmi_finds_the_first_member() {
        if !SimdBackend::Avx512Vbmi.is_available() {
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
            let actual = unsafe { bitset_vbmi::<false>(&bits, &haystack) };
            assert_eq!(actual, 17, "member={member}");
        }
    }

    #[test]
    fn bitset_vbmi_finds_the_first_nonmember() {
        if !SimdBackend::Avx512Vbmi.is_available() {
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
            let actual = unsafe { bitset_vbmi::<true>(&bits, &haystack) };
            assert_eq!(actual, 17, "member={member}");
        }
    }

    #[test]
    fn rbitset_vbmi_finds_the_last_member() {
        if !SimdBackend::Avx512Vbmi.is_available() {
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
            let actual = unsafe { rbitset_vbmi::<false>(&bits, &haystack) };
            assert_eq!(actual, 96, "member={member}");
        }
    }

    #[test]
    fn rbitset_vbmi_finds_the_last_nonmember() {
        if !SimdBackend::Avx512Vbmi.is_available() {
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
            let actual = unsafe { rbitset_vbmi::<true>(&bits, &haystack) };
            assert_eq!(actual, 96, "member={member}");
        }
    }
}
