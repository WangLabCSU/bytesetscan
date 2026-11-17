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
fn scan(haystack: &[u8], classify: impl Fn(v128) -> u16) -> usize {
    // Scan an initial unaligned vector,
    // aligned one-vector loop, then the overlapping suffix.
    // SAFETY: this module is compiled only with simd128 enabled.
    // Inputs below eight bytes use initialized local storage; 8..15 bytes use
    // two overlapping eight-byte loads. Full-vector loads stay inside haystack.
    unsafe {
        let ptr = haystack.as_ptr();
        let len = haystack.len();
        // Fewer than eight bytes cannot fill either overlapping load.
        if len < 8 {
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
        if len < 16 {
            // SAFETY: the preceding branch establishes 8 <= len < 16.
            // classify returns one selected bit per byte, in input order.
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
fn rscan(haystack: &[u8], classify: impl Fn(v128) -> u16) -> usize {
    // Scan an initial unaligned vector,
    // aligned one-vector loop, then the overlapping prefix.
    // SAFETY: this module is compiled only with simd128 enabled.
    // Inputs below eight bytes use initialized local storage; 8..15 bytes use
    // two overlapping eight-byte loads. Full-vector loads stay inside haystack.
    unsafe {
        let ptr = haystack.as_ptr();
        let len = haystack.len();
        // Fewer than eight bytes cannot fill either overlapping load.
        if len < 8 {
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
        if len < 16 {
            // SAFETY: the preceding branch establishes 8 <= len < 16.
            // classify returns one selected bit per byte, in input order.
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
    use super::{i8x16_bitmask, mask16_overlap, rscan, scan, u8x16_eq, u8x16_splat};

    #[test]
    fn scan_returns_the_first_selected_byte_in_overlapping_input() {
        let mut flags = [0u8; 15];
        flags[3] = 0b11111111;
        flags[12] = 0b11111111;
        assert_eq!(scan(&flags, |bytes| i8x16_bitmask(bytes)), 3);
    }

    #[test]
    fn rscan_returns_the_last_selected_byte_in_overlapping_input() {
        let mut flags = [0u8; 15];
        flags[3] = 0b11111111;
        flags[12] = 0b11111111;
        assert_eq!(rscan(&flags, |bytes| i8x16_bitmask(bytes)), 12);
    }

    #[test]
    fn mask16_overlap_preserves_each_selected_bytes_position() {
        for len in 8..16 {
            for position in 0..len {
                let mut flags = [0u8; 15];
                flags[position] = 0b11111111;
                // SAFETY: each slice has 8..15 bytes; bitmask extracts one
                // flag from each byte independently, preserving lane order.
                let mask = unsafe { mask16_overlap(&flags[..len], |bytes| i8x16_bitmask(bytes)) };
                assert_eq!(mask, 1 << position, "len={len}, byte={position}");
            }
        }
    }

    #[test]
    fn mask16_overlap_clears_bits_past_the_input() {
        // Fifteen selected bytes must set only bits 0..14, even though
        // the classifier sees sixteen selected lanes before overlap is merged.
        let flags = [0b11111111; 15];
        // SAFETY: the slice has fifteen bytes and bitmask preserves lane order.
        let mask = unsafe { mask16_overlap(&flags, |bytes| i8x16_bitmask(bytes)) };
        assert_eq!(mask, 0b0111_1111_1111_1111);
    }

    #[test]
    fn scan_excludes_matching_padding() {
        // No input byte is zero, but every padding byte is. Empty input
        // must also report its end, even though all sixteen lanes match.
        for len in 0..8 {
            let haystack = [1u8; 7];
            let actual = scan(&haystack[..len], |bytes| {
                i8x16_bitmask(u8x16_eq(bytes, u8x16_splat(0)))
            });
            assert_eq!(actual, len);
        }
    }

    #[test]
    fn rscan_excludes_matching_padding() {
        for len in 0..8 {
            let haystack = [1u8; 7];
            let actual = rscan(&haystack[..len], |bytes| {
                i8x16_bitmask(u8x16_eq(bytes, u8x16_splat(0)))
            });
            assert_eq!(actual, len);
        }
    }
}
