// Byte-table lookup reference:
// https://doc.rust-lang.org/core/arch/wasm32/fn.u8x16_swizzle.html

use core::arch::wasm32::*;

pub(super) fn search<const REVERT: bool>(
    low: &[u8; 16],
    high: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // Use scalar matching when the input cannot fill a 16-byte vector.
    if haystack.len() < 16 {
        return super::search_fallback::<REVERT>(low, high, haystack);
    }
    // SAFETY: dispatch establishes SIMD support; the check above ensures
    // at least 16 input bytes before entering the vector scan.
    // Table loads and every scan load below stay within initialized slices.
    unsafe {
        // Load all sixteen table entries into sixteen u8 lanes. No nibble
        // selection happens during this load; the following lookup uses them.
        let low = v128_load(low.as_ptr().cast());
        // Load all sixteen table entries into sixteen u8 lanes. No nibble
        // selection happens during this load; the following lookup uses them.
        let high = v128_load(high.as_ptr().cast());
        scan(haystack, |bytes| {
            // Classify each input byte independently. The outer bitwise AND
            // intersects the two lookup results; a nonzero byte means membership.
            let flags = v128_and(
                // Broadcast 0b00001111 into all byte lanes; AND extracts b & 0b00001111.
                // The byte lookup returns low[b & 0b00001111] for each independent lane.
                // All indices here are in 0..16. The lookup's out-of-range
                // zeroing behavior therefore does not participate in classification.
                u8x16_swizzle(low, v128_and(bytes, u8x16_splat(0b00001111))),
                // Logical right shift acts on individual u8 lanes: each index is
                // b >> 4, with no spill from neighboring bytes. Look up high[index].
                // All indices here are in 0..16. The lookup's out-of-range
                // zeroing behavior therefore does not participate in classification.
                u8x16_swizzle(high, u8x16_shr(bytes, 4)),
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
    // SAFETY: SIMD support is established by the entry point along with len >= 16.
    // All full input loads remain inside haystack.
    unsafe {
        let base = haystack.as_ptr();
        let len = haystack.len();
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
    low: &[u8; 16],
    high: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // Use scalar matching when the input cannot fill a 16-byte vector.
    if haystack.len() < 16 {
        return super::rsearch_fallback::<REVERT>(low, high, haystack);
    }
    // SAFETY: dispatch establishes SIMD support; the check above ensures
    // at least 16 input bytes before entering the vector scan.
    // Table loads and every scan load below stay within initialized slices.
    unsafe {
        // Load all sixteen table entries into sixteen u8 lanes. No nibble
        // selection happens during this load; the following lookup uses them.
        let low = v128_load(low.as_ptr().cast());
        // Load all sixteen table entries into sixteen u8 lanes. No nibble
        // selection happens during this load; the following lookup uses them.
        let high = v128_load(high.as_ptr().cast());
        rscan(haystack, |bytes| {
            // Classify each input byte independently. The outer bitwise AND
            // intersects the two lookup results; a nonzero byte means membership.
            let flags = v128_and(
                // Broadcast 0b00001111 into all byte lanes; AND extracts b & 0b00001111.
                // The byte lookup returns low[b & 0b00001111] for each independent lane.
                // All indices here are in 0..16. The lookup's out-of-range
                // zeroing behavior therefore does not participate in classification.
                u8x16_swizzle(low, v128_and(bytes, u8x16_splat(0b00001111))),
                // Logical right shift acts on individual u8 lanes: each index is
                // b >> 4, with no spill from neighboring bytes. Look up high[index].
                // All indices here are in 0..16. The lookup's out-of-range
                // zeroing behavior therefore does not participate in classification.
                u8x16_swizzle(high, u8x16_shr(bytes, 4)),
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
    // SAFETY: SIMD support is established by the entry point along with len >= 16.
    // All full input loads remain inside haystack.
    unsafe {
        let base = haystack.as_ptr();
        let len = haystack.len();
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
