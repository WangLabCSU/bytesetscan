use core::arch::wasm32::*;

use super::{rscan, rsearch_fallback, scan, search_fallback};

pub(super) fn search<const REVERT: bool>(bits: &[u64; 4], haystack: &[u8]) -> usize {
    if haystack.len() < 16 {
        return search_fallback::<REVERT>(bits, haystack);
    }
    // SAFETY: this module is compiled only with simd128 enabled. The table
    // occupies 32 initialized bytes, and the scanner bounds each input load.
    unsafe {
        // WebAssembly is little-endian, so memory byte i holds membership
        // bits for values 8*i..8*i+7 directly from the original bitmap.
        let ptr = bits.as_ptr().cast::<u8>();
        let lower = v128_load(ptr.cast());
        let upper = v128_load(ptr.add(16).cast());
        let bit_masks = u8x16(
            0b00000001, 0b00000010, 0b00000100, 0b00001000, 0b00010000, 0b00100000, 0b01000000,
            0b10000000, 0b00000001, 0b00000010, 0b00000100, 0b00001000, 0b00010000, 0b00100000,
            0b01000000, 0b10000000,
        );
        scan::<16>(haystack, |ptr| {
            // v128_load accepts unaligned addresses and keeps byte lane order.
            mask16::<REVERT>(lower, upper, bit_masks, v128_load(ptr.cast())) as u64
        })
    }
}

pub(super) fn rsearch<const REVERT: bool>(bits: &[u64; 4], haystack: &[u8]) -> usize {
    if haystack.len() < 16 {
        return rsearch_fallback::<REVERT>(bits, haystack);
    }
    // SAFETY: this module is compiled only with simd128 enabled. The table
    // occupies 32 initialized bytes, and the scanner bounds each input load.
    unsafe {
        // WebAssembly is little-endian, so memory byte i holds membership
        // bits for values 8*i..8*i+7 directly from the original bitmap.
        let ptr = bits.as_ptr().cast::<u8>();
        let lower = v128_load(ptr.cast());
        let upper = v128_load(ptr.add(16).cast());
        let bit_masks = u8x16(
            0b00000001, 0b00000010, 0b00000100, 0b00001000, 0b00010000, 0b00100000, 0b01000000,
            0b10000000, 0b00000001, 0b00000010, 0b00000100, 0b00001000, 0b00010000, 0b00100000,
            0b01000000, 0b10000000,
        );
        rscan::<16>(haystack, |ptr| {
            // v128_load accepts unaligned addresses and keeps byte lane order.
            mask16::<REVERT>(lower, upper, bit_masks, v128_load(ptr.cast())) as u64
        })
    }
}

#[inline(always)]
fn mask16<const REVERT: bool>(lower: v128, upper: v128, bit_masks: v128, bytes: v128) -> u16 {
    // Shift unsigned byte lanes independently: every index is b >> 3.
    let indices = u8x16_shr(bytes, 3);
    // Swizzle returns zero for indices outside 0..15. Subtracting 16 wraps
    // indices 0..15 to 240..255, so exactly one of these lookups contributes.
    let entry = v128_or(
        u8x16_swizzle(lower, indices),
        u8x16_swizzle(upper, u8x16_sub(indices, u8x16_splat(16))),
    );
    // b & 7 selects the single bit representing this byte in its table entry.
    let bit = u8x16_swizzle(bit_masks, v128_and(bytes, u8x16_splat(0b00000111)));
    let absent = u8x16_eq(v128_and(entry, bit), u8x16_splat(0));
    // Bitmask extracts the highest bit of each comparison lane in input order.
    // Its u16 type limits inversion to the sixteen valid lane positions.
    let absent = i8x16_bitmask(absent);
    if REVERT {
        absent
    } else {
        !absent
    }
}
