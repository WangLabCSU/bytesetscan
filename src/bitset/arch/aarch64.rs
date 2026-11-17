use core::arch::aarch64::*;

use super::{rscan, rsearch_fallback, scan, search_fallback};

#[target_feature(enable = "neon")]
pub(super) fn search<const REVERT: bool>(bits: &[u64; 4], haystack: &[u8]) -> usize {
    if haystack.len() < 16 {
        return search_fallback::<REVERT>(bits, haystack);
    }
    // SAFETY: dispatch checked NEON support; the local table holds 32 bytes
    // and the scanner supplies only complete, in-bounds 16-byte input blocks.
    unsafe {
        // A word's least-significant bit represents its lowest byte value.
        // Convert each word to little-endian memory order before byte loads,
        // including on big-endian AArch64, so table[b >> 3] has the same
        // meaning as the scalar word lookup.
        let table = bits.map(u64::to_le);
        let ptr = table.as_ptr().cast::<u8>();
        let table = uint8x16x2_t(vld1q_u8(ptr), vld1q_u8(ptr.add(16)));
        scan::<16>(haystack, |ptr| {
            // vld1q accepts unaligned addresses; every input lane is initialized.
            mask16::<REVERT>(table, vld1q_u8(ptr)) as u64
        })
    }
}

#[target_feature(enable = "neon")]
pub(super) fn rsearch<const REVERT: bool>(bits: &[u64; 4], haystack: &[u8]) -> usize {
    if haystack.len() < 16 {
        return rsearch_fallback::<REVERT>(bits, haystack);
    }
    // SAFETY: dispatch checked NEON support; the local table holds 32 bytes
    // and the scanner supplies only complete, in-bounds 16-byte input blocks.
    unsafe {
        // A word's least-significant bit represents its lowest byte value.
        // Convert each word to little-endian memory order before byte loads,
        // including on big-endian AArch64, so table[b >> 3] has the same
        // meaning as the scalar word lookup.
        let table = bits.map(u64::to_le);
        let ptr = table.as_ptr().cast::<u8>();
        let table = uint8x16x2_t(vld1q_u8(ptr), vld1q_u8(ptr.add(16)));
        rscan::<16>(haystack, |ptr| {
            // vld1q accepts unaligned addresses; every input lane is initialized.
            mask16::<REVERT>(table, vld1q_u8(ptr)) as u64
        })
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn mask16<const REVERT: bool>(table: uint8x16x2_t, bytes: uint8x16_t) -> u16 {
    // Unlike the x86 lane shift, this shifts each byte independently. b >> 3
    // selects one of the 32 bitset bytes through TBL's two-register lookup.
    let entry = vqtbl2q_u8(table, vshrq_n_u8::<3>(bytes));
    // Mask to 0..7, reinterpret those nonnegative counts as signed, and shift
    // one in every lane. This constructs 1 << (b & 7) without a lookup table.
    let shifts = vreinterpretq_s8_u8(vandq_u8(bytes, vdupq_n_u8(0b00000111)));
    let bit = vshlq_u8(vdupq_n_u8(1), shifts);
    // TST produces an all-one byte exactly when entry & bit is nonzero.
    let mut selected = vtstq_u8(entry, bit);
    if REVERT {
        selected = vmvnq_u8(selected);
    }
    // Weight flag i in each eight-byte half by 1 << i, then sum them.
    // The two sums give bits 0..7 and 8..15 without relying on endian-specific
    // byte-to-u64 reinterpretation. Their maximum sum is 255.
    // SAFETY: the local array contains all eight initialized weight bytes.
    let weights = unsafe {
        vld1_u8(
            [
                1 << 0,
                1 << 1,
                1 << 2,
                1 << 3,
                1 << 4,
                1 << 5,
                1 << 6,
                1 << 7,
            ]
            .as_ptr(),
        )
    };
    let lower = vaddv_u8(vand_u8(vget_low_u8(selected), weights));
    let upper = vaddv_u8(vand_u8(vget_high_u8(selected), weights));
    u16::from(lower) | (u16::from(upper) << 8)
}
