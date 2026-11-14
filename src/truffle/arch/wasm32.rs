// Byte-table lookup reference:
// https://doc.rust-lang.org/core/arch/wasm32/fn.u8x16_swizzle.html

use core::arch::wasm32::*;

pub(super) fn search<const REVERT: bool>(
    lower: &[u8; 16],
    upper: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // SAFETY: dispatch establishes SIMD support. scan/rscan handle short inputs.
    // Table loads and every scan load below stay within initialized slices.
    unsafe {
        // Load all sixteen table entries into sixteen u8 lanes. No nibble
        // selection happens during this load; the following lookup uses them.
        let lower = v128_load(lower.as_ptr().cast());
        // Load all sixteen table entries into sixteen u8 lanes. No nibble
        // selection happens during this load; the following lookup uses them.
        let upper = v128_load(upper.as_ptr().cast());
        // Load two copies of [1 << 0, ..., 1 << 7]. The high nibble h
        // selects the membership bit 1 << (h & 0b00000111).
        let bits = v128_load(
            [
                1u8 << 0,
                1 << 1,
                1 << 2,
                1 << 3,
                1 << 4,
                1 << 5,
                1 << 6,
                1 << 7,
                1 << 0,
                1 << 1,
                1 << 2,
                1 << 3,
                1 << 4,
                1 << 5,
                1 << 6,
                1 << 7,
            ]
            .as_ptr()
            .cast(),
        );
        scan(haystack, |bytes| {
            // The outer AND tests a single membership bit in a table entry.
            // The table half, its entry, and that bit are selected separately below.
            let flags = v128_and(
                // bitselect(yes, no, mask) chooses bits from yes where mask is 1,
                // otherwise from no. This operand order differs from NEON vbsl.
                v128_bitselect(
                    // Broadcast 0b00001111 and AND with each byte; use b & 0b00001111 to look up
                    // the lower-half table entry. The selection mask may discard it.
                    // All indices here are in 0..16. The lookup's out-of-range
                    // zeroing behavior therefore does not participate in classification.
                    u8x16_swizzle(lower, v128_and(bytes, u8x16_splat(0b00001111))),
                    // Use the same low-nibble index in the upper-half table. Unlike
                    // x86 PSHUFB, this lookup does not use bit 7 to choose the half.
                    // All indices here are in 0..16. The lookup's out-of-range
                    // zeroing behavior therefore does not participate in classification.
                    u8x16_swizzle(upper, v128_and(bytes, u8x16_splat(0b00001111))),
                    // Broadcast 0b10000000 and compare UNSIGNED bytes: lanes below 128 become
                    // 0b11111111, others 0b00000000. This selects lower for b < 128, upper
                    // otherwise.
                    u8x16_lt(bytes, u8x16_splat(0b10000000)),
                ),
                // Shift each unsigned byte right by 4, independently of its
                // neighbors. Lookup in bits converts h = b >> 4 into 0b00000001 << (h &
                // 0b00000111).
                // AND with the chosen table entry tests exactly that membership bit.
                // All indices here are in 0..16. The lookup's out-of-range
                // zeroing behavior therefore does not participate in classification.
                u8x16_swizzle(bits, u8x16_shr(bytes, 4)),
            );
            // Membership flags are zero for nonmembers and nonzero for members.
            // Each comparison produces 0b11111111 for a selected byte and
            // 0b00000000 otherwise, so bitmask can extract its top bit.
            let zero = u8x16_splat(0b00000000);
            let selected = if REVERT {
                u8x16_eq(flags, zero) // mismatch: select nonmembers.
            } else {
                u8x16_ne(flags, zero) // find: select members.
            };
            // Pack the top bit of byte lane i into scalar bit i. For 0b11111111/0b00000000
            // flags this is a sixteen-bit result with one bit per input byte.
            i8x16_bitmask(selected)
        })
    }
}

#[inline(always)]
unsafe fn scan(haystack: &[u8], classify: impl Fn(v128) -> u16) -> usize {
    // Scan an initial unaligned vector,
    // aligned one-vector loop, then the overlapping suffix.
    // SAFETY: SIMD support is established by the entry point.
    // Short inputs use initialized local storage. All full input loads remain inside haystack.
    unsafe {
        let base = haystack.as_ptr();
        let len = haystack.len();
        // truffleMini/truffleRevMini: zero padding is excluded after classification.
        if len < 16 {
            let mut bytes = [0u8; 16];
            bytes[..len].copy_from_slice(haystack);
            // Read sixteen consecutive input bytes into a vector; this load
            // does not require a 16-byte-aligned address. The scan bounds or
            // initialized short-input buffer guarantee all sixteen bytes exist.
            let mask = classify(v128_load(bytes.as_ptr().cast()))
                & u16::MAX.checked_shr((16 - len) as u32).unwrap_or(0);
            return if mask != 0 {
                // The lowest set bit identifies the first selected lane.
                mask.trailing_zeros() as usize
            } else {
                len
            };
        }
        // Read sixteen consecutive input bytes into a vector; this load
        // does not require a 16-byte-aligned address. The scan bounds or
        // initialized short-input buffer guarantee all sixteen bytes exist.
        let mask = classify(v128_load(base.cast()));
        if mask != 0 {
            // The lowest set bit identifies the first selected lane.
            return mask.trailing_zeros() as usize;
        }
        // Masking the low four address bits gives the remainder modulo 16.
        // 16 - remainder is the distance to the next 16-byte boundary
        // (16 bytes if already aligned). The initial block was checked,
        // so any overlap contains only rejected bytes.
        let mut start = 16 - (base as usize & 0b00001111);
        let last = len - 16;
        while start < last {
            // Read sixteen consecutive input bytes into a vector; this load
            // does not require a 16-byte-aligned address. The scan bounds or
            // initialized short-input buffer guarantee all sixteen bytes exist.
            let mask = classify(v128_load(base.add(start).cast()));
            if mask != 0 {
                // The lowest set bit identifies the first selected lane.
                // Add this block's start to return a position in the original input.
                return start + mask.trailing_zeros() as usize;
            }
            start += 16;
        }
        // Read sixteen consecutive input bytes into a vector; this load
        // does not require a 16-byte-aligned address. The scan bounds or
        // initialized short-input buffer guarantee all sixteen bytes exist.
        let mask = classify(v128_load(base.add(last).cast()));
        if mask != 0 {
            // The lowest set bit identifies the first selected lane.
            // Add this block's start to return a position in the original input.
            last + mask.trailing_zeros() as usize
        } else {
            len
        }
    }
}

pub(super) fn rsearch<const REVERT: bool>(
    lower: &[u8; 16],
    upper: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // SAFETY: dispatch establishes SIMD support. scan/rscan handle short inputs.
    // Table loads and every scan load below stay within initialized slices.
    unsafe {
        // Load all sixteen table entries into sixteen u8 lanes. No nibble
        // selection happens during this load; the following lookup uses them.
        let lower = v128_load(lower.as_ptr().cast());
        // Load all sixteen table entries into sixteen u8 lanes. No nibble
        // selection happens during this load; the following lookup uses them.
        let upper = v128_load(upper.as_ptr().cast());
        // Load two copies of [1 << 0, ..., 1 << 7]. The high nibble h
        // selects the membership bit 1 << (h & 0b00000111).
        let bits = v128_load(
            [
                1u8 << 0,
                1 << 1,
                1 << 2,
                1 << 3,
                1 << 4,
                1 << 5,
                1 << 6,
                1 << 7,
                1 << 0,
                1 << 1,
                1 << 2,
                1 << 3,
                1 << 4,
                1 << 5,
                1 << 6,
                1 << 7,
            ]
            .as_ptr()
            .cast(),
        );
        rscan(haystack, |bytes| {
            // The outer AND tests a single membership bit in a table entry.
            // The table half, its entry, and that bit are selected separately below.
            let flags = v128_and(
                // bitselect(yes, no, mask) chooses bits from yes where mask is 1,
                // otherwise from no. This operand order differs from NEON vbsl.
                v128_bitselect(
                    // Broadcast 0b00001111 and AND with each byte; use b & 0b00001111 to look up
                    // the lower-half table entry. The selection mask may discard it.
                    // All indices here are in 0..16. The lookup's out-of-range
                    // zeroing behavior therefore does not participate in classification.
                    u8x16_swizzle(lower, v128_and(bytes, u8x16_splat(0b00001111))),
                    // Use the same low-nibble index in the upper-half table. Unlike
                    // x86 PSHUFB, this lookup does not use bit 7 to choose the half.
                    // All indices here are in 0..16. The lookup's out-of-range
                    // zeroing behavior therefore does not participate in classification.
                    u8x16_swizzle(upper, v128_and(bytes, u8x16_splat(0b00001111))),
                    // Broadcast 0b10000000 and compare UNSIGNED bytes: lanes below 128 become
                    // 0b11111111, others 0b00000000. This selects lower for b < 128, upper
                    // otherwise.
                    u8x16_lt(bytes, u8x16_splat(0b10000000)),
                ),
                // Shift each unsigned byte right by 4, independently of its
                // neighbors. Lookup in bits converts h = b >> 4 into 0b00000001 << (h &
                // 0b00000111).
                // AND with the chosen table entry tests exactly that membership bit.
                // All indices here are in 0..16. The lookup's out-of-range
                // zeroing behavior therefore does not participate in classification.
                u8x16_swizzle(bits, u8x16_shr(bytes, 4)),
            );
            // Membership flags are zero for nonmembers and nonzero for members.
            // Each comparison produces 0b11111111 for a selected byte and
            // 0b00000000 otherwise, so bitmask can extract its top bit.
            let zero = u8x16_splat(0b00000000);
            let selected = if REVERT {
                u8x16_eq(flags, zero) // mismatch: select nonmembers.
            } else {
                u8x16_ne(flags, zero) // find: select members.
            };
            // Pack the top bit of byte lane i into scalar bit i. For 0b11111111/0b00000000
            // flags this is a sixteen-bit result with one bit per input byte.
            i8x16_bitmask(selected)
        })
    }
}

#[inline(always)]
unsafe fn rscan(haystack: &[u8], classify: impl Fn(v128) -> u16) -> usize {
    // Scan an initial unaligned vector,
    // aligned one-vector loop, then the overlapping prefix.
    // SAFETY: SIMD support is established by the entry point.
    // Short inputs use initialized local storage. All full input loads remain inside haystack.
    unsafe {
        let base = haystack.as_ptr();
        let len = haystack.len();
        // truffleMini/truffleRevMini: zero padding is excluded after classification.
        if len < 16 {
            let mut bytes = [0u8; 16];
            bytes[..len].copy_from_slice(haystack);
            // Read sixteen consecutive input bytes into a vector; this load
            // does not require a 16-byte-aligned address. The scan bounds or
            // initialized short-input buffer guarantee all sixteen bytes exist.
            let mask = classify(v128_load(bytes.as_ptr().cast()))
                & u16::MAX.checked_shr((16 - len) as u32).unwrap_or(0);
            return if mask != 0 {
                // 15 - leading_zeros locates the highest set bit in this u16,
                // which identifies the last selected byte lane.
                15 - mask.leading_zeros() as usize
            } else {
                len
            };
        }
        let last = len - 16;
        // Read sixteen consecutive input bytes into a vector; this load
        // does not require a 16-byte-aligned address. The scan bounds or
        // initialized short-input buffer guarantee all sixteen bytes exist.
        let mask = classify(v128_load(base.add(last).cast()));
        if mask != 0 {
            // 15 - leading_zeros locates the highest set bit in this u16,
            // which identifies the last selected byte lane.
            // Add this block's start to return a position in the original input.
            return last + 15 - mask.leading_zeros() as usize;
        }
        // The end address's low four bits give its remainder modulo 16.
        // Subtract that remainder from len to round base + end down to a
        // 16-byte boundary. The initial suffix scan covered the skipped bytes.
        let mut end = len - (base.add(len) as usize & 0b00001111);
        while end > 16 {
            end -= 16;
            // Read sixteen consecutive input bytes into a vector; this load
            // does not require a 16-byte-aligned address. The scan bounds or
            // initialized short-input buffer guarantee all sixteen bytes exist.
            let mask = classify(v128_load(base.add(end).cast()));
            if mask != 0 {
                // 15 - leading_zeros locates the highest set bit in this u16,
                // which identifies the last selected byte lane.
                // Add this block's start to return a position in the original input.
                return end + 15 - mask.leading_zeros() as usize;
            }
        }
        // Read sixteen consecutive input bytes into a vector; this load
        // does not require a 16-byte-aligned address. The scan bounds or
        // initialized short-input buffer guarantee all sixteen bytes exist.
        let mask = classify(v128_load(base.cast()));
        if mask != 0 {
            // 15 - leading_zeros locates the highest set bit in this u16,
            // which identifies the last selected byte lane.
            15 - mask.leading_zeros() as usize
        } else {
            len
        }
    }
}
