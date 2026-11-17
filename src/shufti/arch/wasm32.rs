// Byte-table lookup reference:
// https://doc.rust-lang.org/core/arch/wasm32/fn.u8x16_swizzle.html

use core::arch::wasm32::*;

pub(super) fn search<const REVERT: bool>(
    low: &[u8; 16],
    high: &[u8; 16],
    haystack: &[u8],
) -> usize {
    if haystack.len() < 8 {
        return super::search_fallback::<REVERT>(low, high, haystack);
    }
    // SAFETY: dispatch establishes SIMD support; the check above ensures
    // at least eight input bytes before entering the vector scan.
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
    // SAFETY: the entry point establishes SIMD support and len >= 8.
    // Short inputs use two overlapping eight-byte loads; full loads stay in bounds.
    unsafe {
        let ptr = haystack.as_ptr();
        let len = haystack.len();
        if len < 16 {
            // SAFETY: 8 <= len < 16; classify returns independent byte flags.
            let mask = mask16_overlap(haystack, classify);
            return if mask != 0 {
                mask.trailing_zeros() as usize
            } else {
                len
            };
        }
        // Read sixteen consecutive input bytes into a vector; this load
        // does not require a 16-byte-aligned address. The scan bounds or
        // initialized short-input buffer guarantee all sixteen bytes exist.
        let mask = classify(v128_load(ptr.cast()));
        if mask != 0 {
            // The lowest set bit identifies the first selected lane.
            return mask.trailing_zeros() as usize;
        }
        // Masking the low four address bits gives the remainder modulo 16.
        // 16 - remainder is the distance to the next 16-byte boundary
        // (16 bytes if already aligned). The initial block was checked,
        // so any overlap contains only rejected bytes.
        let mut start = 16 - (ptr as usize & 0b00001111);
        let last = len - 16;
        while start < last {
            // Read sixteen consecutive input bytes into a vector; this load
            // does not require a 16-byte-aligned address. The scan bounds or
            // initialized short-input buffer guarantee all sixteen bytes exist.
            let mask = classify(v128_load(ptr.add(start).cast()));
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
        let mask = classify(v128_load(ptr.add(last).cast()));
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
    if haystack.len() < 8 {
        return super::rsearch_fallback::<REVERT>(low, high, haystack);
    }
    // SAFETY: dispatch establishes SIMD support; the check above ensures
    // at least eight input bytes before entering the vector scan.
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
    // SAFETY: the entry point establishes SIMD support and len >= 8.
    // Short inputs use two overlapping eight-byte loads; full loads stay in bounds.
    unsafe {
        let ptr = haystack.as_ptr();
        let len = haystack.len();
        if len < 16 {
            // SAFETY: 8 <= len < 16; classify returns independent byte flags.
            let mask = mask16_overlap(haystack, classify);
            return if mask != 0 {
                15 - mask.leading_zeros() as usize
            } else {
                len
            };
        }
        let last = len - 16;
        // Read sixteen consecutive input bytes into a vector; this load
        // does not require a 16-byte-aligned address. The scan bounds or
        // initialized short-input buffer guarantee all sixteen bytes exist.
        let mask = classify(v128_load(ptr.add(last).cast()));
        if mask != 0 {
            // 15 - leading_zeros locates the highest set bit in this u16,
            // which identifies the last selected byte lane.
            // Add this block's start to return a position in the original input.
            return last + 15 - mask.leading_zeros() as usize;
        }
        // The end address's low four bits give its remainder modulo 16.
        // Subtract that remainder from len to round ptr + end down to a
        // 16-byte boundary. The initial suffix scan covered the skipped bytes.
        let mut end = len - (ptr.add(len) as usize & 0b00001111);
        while end > 16 {
            end -= 16;
            // Read sixteen consecutive input bytes into a vector; this load
            // does not require a 16-byte-aligned address. The scan bounds or
            // initialized short-input buffer guarantee all sixteen bytes exist.
            let mask = classify(v128_load(ptr.add(end).cast()));
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
        let mask = classify(v128_load(ptr.cast()));
        if mask != 0 {
            // 15 - leading_zeros locates the highest set bit in this u16,
            // which identifies the last selected byte lane.
            15 - mask.leading_zeros() as usize
        } else {
            len
        }
    }
}

/// Returns one selected bit per input byte, with higher bits clear.
///
/// # Safety
/// `haystack` must contain 8 to 15 bytes. `classify` must classify each byte
/// independently and return its flag in the corresponding bit of the mask.
#[inline(always)]
unsafe fn mask16_overlap(haystack: &[u8], classify: impl Fn(v128) -> u16) -> u16 {
    // SAFETY: this module requires simd128. Both loads read exactly eight
    // bytes inside haystack, without requiring alignment. The second load
    // replaces the zero-filled upper half, so all sixteen lanes are initialized.
    unsafe {
        let ptr = haystack.as_ptr();
        let last = haystack.len() - 8;
        let prefix = v128_load64_zero(ptr.cast());
        let bytes = v128_load64_lane::<1>(prefix, ptr.add(last).cast());
        // For len = 15 the vector contains input positions
        // [0, ..., 7 | 7, ..., 14]. Classify both halves in one lookup.
        let mask = classify(bytes);
        // Bits 0..7 already describe the prefix. Move bits 8..15 down to
        // positions last..len; duplicate input bytes have identical flags,
        // so OR merges them. No resulting bit lies at or above len.
        (mask & 0b1111_1111) | ((mask >> 8) << last)
    }
}

#[cfg(test)]
mod tests {
    use super::{i8x16_bitmask, mask16_overlap, rscan, scan};

    #[test]
    fn scan_returns_the_first_selected_byte_in_overlapping_input() {
        let mut flags = [0u8; 15];
        flags[3] = 0b11111111;
        flags[12] = 0b11111111;
        // SAFETY: this module requires simd128 and the input has fifteen bytes.
        let actual = unsafe { scan(&flags, |bytes| i8x16_bitmask(bytes)) };
        assert_eq!(actual, 3);
    }

    #[test]
    fn rscan_returns_the_last_selected_byte_in_overlapping_input() {
        let mut flags = [0u8; 15];
        flags[3] = 0b11111111;
        flags[12] = 0b11111111;
        // SAFETY: this module requires simd128 and the input has fifteen bytes.
        let actual = unsafe { rscan(&flags, |bytes| i8x16_bitmask(bytes)) };
        assert_eq!(actual, 12);
    }

    #[test]
    fn mask16_overlap_preserves_each_selected_bytes_position() {
        for len in 8..16 {
            for position in 0..len {
                let mut flags = [0u8; 15];
                flags[position] = 0b11111111;
                // SAFETY: each slice has 8..15 bytes; bitmask extracts one
                // flag per byte independently, preserving lane order.
                let mask = unsafe { mask16_overlap(&flags[..len], |bytes| i8x16_bitmask(bytes)) };
                assert_eq!(mask, 1 << position, "len={len}, byte={position}");
            }
        }
    }

    #[test]
    fn mask16_overlap_clears_bits_past_the_input() {
        // Fifteen selected bytes set bits 0..14, despite sixteen selected lanes.
        let flags = [0b11111111; 15];
        // SAFETY: the slice has fifteen bytes and bitmask preserves lane order.
        let mask = unsafe { mask16_overlap(&flags, |bytes| i8x16_bitmask(bytes)) };
        assert_eq!(mask, 0b0111_1111_1111_1111);
    }
}
