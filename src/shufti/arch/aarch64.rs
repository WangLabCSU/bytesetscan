// Narrowing instruction reference:
// https://doc.rust-lang.org/core/arch/aarch64/fn.vshrn_n_u16.html

use core::arch::aarch64::*;

#[target_feature(enable = "neon")]
pub(super) unsafe fn search<const REVERT: bool>(
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
        let low = vld1q_u8(low.as_ptr());
        // Load all sixteen table entries into sixteen u8 lanes. No nibble
        // selection happens during this load; the following lookup uses them.
        let high = vld1q_u8(high.as_ptr());
        scan(haystack, |bytes| {
            // Classify each input byte independently. The outer bitwise AND
            // intersects the two lookup results; a nonzero byte means membership.
            let flags = vandq_u8(
                // Broadcast 0b00001111 into all byte lanes; AND extracts b & 0b00001111.
                // The byte lookup returns low[b & 0b00001111] for each independent lane.
                // All indices here are in 0..16. The lookup's out-of-range
                // zeroing behavior therefore does not participate in classification.
                vqtbl1q_u8(low, vandq_u8(bytes, vdupq_n_u8(0b00001111))),
                // Logical right shift acts on individual u8 lanes: each index is
                // b >> 4, with no spill from neighboring bytes. Look up high[index].
                // All indices here are in 0..16. The lookup's out-of-range
                // zeroing behavior therefore does not participate in classification.
                vqtbl1q_u8(high, vshrq_n_u8::<4>(bytes)),
            );
            // Broadcast zero and compare each u8 lane: no membership flag
            // becomes 0b11111111, while a member becomes 0b00000000. These are nonmember flags.
            let mut selected = vceqq_u8(flags, vdupq_n_u8(0b00000000));
            if !REVERT {
                // Bitwise NOT switches nonmember flags into member flags for find;
                // mismatch leaves the comparison result unchanged.
                selected = vmvnq_u8(selected);
            }
            selected_mask(selected)
        })
    }
}

#[inline(always)]
unsafe fn scan(haystack: &[u8], classify: impl Fn(uint8x16_t) -> u64) -> usize {
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
        let mask = classify(vld1q_u8(base));
        if mask != 0 {
            // The lowest set bit identifies the first selected lane.
            // Each lane occupies four mask bits, so divide that bit index by 4.
            return mask.trailing_zeros() as usize / 4;
        }
        // Advance to the next vector-aligned address after testing the
        // initial unaligned block. Any overlap contains only rejected bytes.
        let mut start = 16 - (base as usize & 0b00001111);
        let last = len - 16;
        while start < last {
            // Read sixteen consecutive input bytes into a vector; this load
            // does not require a 16-byte-aligned address. The scan bounds or
            // initialized short-input buffer guarantee all sixteen bytes exist.
            let mask = classify(vld1q_u8(base.add(start)));
            if mask != 0 {
                // The lowest set bit identifies the first selected lane.
                // Each lane occupies four mask bits, so divide that bit index by 4.
                // Add this block's start to return a position in the original input.
                return start + mask.trailing_zeros() as usize / 4;
            }
            start += 16;
        }
        // Read sixteen consecutive input bytes into a vector; this load
        // does not require a 16-byte-aligned address. The scan bounds or
        // initialized short-input buffer guarantee all sixteen bytes exist.
        let mask = classify(vld1q_u8(base.add(last)));
        if mask != 0 {
            // The lowest set bit identifies the first selected lane.
            // Each lane occupies four mask bits, so divide that bit index by 4.
            // Add this block's start to return a position in the original input.
            last + mask.trailing_zeros() as usize / 4
        } else {
            len
        }
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn rsearch<const REVERT: bool>(
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
        let low = vld1q_u8(low.as_ptr());
        // Load all sixteen table entries into sixteen u8 lanes. No nibble
        // selection happens during this load; the following lookup uses them.
        let high = vld1q_u8(high.as_ptr());
        rscan(haystack, |bytes| {
            // Classify each input byte independently. The outer bitwise AND
            // intersects the two lookup results; a nonzero byte means membership.
            let flags = vandq_u8(
                // Broadcast 0b00001111 into all byte lanes; AND extracts b & 0b00001111.
                // The byte lookup returns low[b & 0b00001111] for each independent lane.
                // All indices here are in 0..16. The lookup's out-of-range
                // zeroing behavior therefore does not participate in classification.
                vqtbl1q_u8(low, vandq_u8(bytes, vdupq_n_u8(0b00001111))),
                // Logical right shift acts on individual u8 lanes: each index is
                // b >> 4, with no spill from neighboring bytes. Look up high[index].
                // All indices here are in 0..16. The lookup's out-of-range
                // zeroing behavior therefore does not participate in classification.
                vqtbl1q_u8(high, vshrq_n_u8::<4>(bytes)),
            );
            // Broadcast zero and compare each u8 lane: no membership flag
            // becomes 0b11111111, while a member becomes 0b00000000. These are nonmember flags.
            let mut selected = vceqq_u8(flags, vdupq_n_u8(0b00000000));
            if !REVERT {
                // Bitwise NOT switches nonmember flags into member flags for find;
                // mismatch leaves the comparison result unchanged.
                selected = vmvnq_u8(selected);
            }
            selected_mask(selected)
        })
    }
}

#[inline(always)]
unsafe fn rscan(haystack: &[u8], classify: impl Fn(uint8x16_t) -> u64) -> usize {
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
        let mask = classify(vld1q_u8(base.add(last)));
        if mask != 0 {
            // 63 - leading_zeros locates the highest set bit in this u64.
            // Divide by 4 to convert its four-bit nibble back to a byte lane.
            // Add this block's start to return a position in the original input.
            return last + (63 - mask.leading_zeros() as usize) / 4;
        }
        // Round the exclusive end down to a vector boundary. The initial
        // suffix scan has already ruled out any bytes skipped by rounding.
        let mut end = len - (base.add(len) as usize & 0b00001111);
        while end > 16 {
            end -= 16;
            // Read sixteen consecutive input bytes into a vector; this load
            // does not require a 16-byte-aligned address. The scan bounds or
            // initialized short-input buffer guarantee all sixteen bytes exist.
            let mask = classify(vld1q_u8(base.add(end)));
            if mask != 0 {
                // 63 - leading_zeros locates the highest set bit in this u64.
                // Divide by 4 to convert its four-bit nibble back to a byte lane.
                // Add this block's start to return a position in the original input.
                return end + (63 - mask.leading_zeros() as usize) / 4;
            }
        }
        // Read sixteen consecutive input bytes into a vector; this load
        // does not require a 16-byte-aligned address. The scan bounds or
        // initialized short-input buffer guarantee all sixteen bytes exist.
        let mask = classify(vld1q_u8(base));
        if mask != 0 {
            // 63 - leading_zeros locates the highest set bit in this u64.
            // Divide by 4 to convert its four-bit nibble back to a byte lane.
            (63 - mask.leading_zeros() as usize) / 4
        } else {
            len
        }
    }
}

// Pack 0b00000000/0b11111111 byte flags into four bits per input byte.
// Byte i always occupies bits 4*i..4*i+3, regardless of target byte order.
#[inline(always)]
unsafe fn selected_mask(selected: uint8x16_t) -> u64 {
    // SAFETY: the caller establishes NEON support. These operations only
    // reinterpret, narrow, and extract values already held in registers.
    let mask = unsafe {
        // Read each pair of flags as a u16, shift right by four, then keep
        // its low byte. Each original flag contributes one 0b0000/0b1111
        // nibble. On little-endian, [0b11111111, 0] becomes 0b00001111;
        // on big-endian, the same pair becomes 0b11110000.
        let packed = vshrn_n_u16::<4>(vreinterpretq_u16_u8(selected));
        vget_lane_u64::<0>(vreinterpret_u64_u8(packed))
    };
    #[cfg(target_endian = "big")]
    {
        // Big-endian puts the first input flag in the highest nibble of
        // this u64. Reverse all bits to restore input order: every nibble
        // contains four identical bits, so reversal preserves its value.
        // Swapping bytes alone would leave each pair of nibbles reversed.
        mask.reverse_bits()
    }
    #[cfg(target_endian = "little")]
    {
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::{rscan, scan, selected_mask, vld1q_u8};
    use crate::SimdBackend;

    #[test]
    fn selected_mask_preserves_every_byte_position() {
        if !SimdBackend::Neon.is_available() {
            return;
        }
        for position in 0..16 {
            let mut flags = [0u8; 16];
            flags[position] = 0b11111111;
            // SAFETY: NEON is available and all sixteen flags are initialized.
            let mask = unsafe { selected_mask(vld1q_u8(flags.as_ptr())) };
            assert_eq!(mask, 0b1111u64 << (position * 4), "byte {position}");
        }
    }

    #[test]
    fn scan_returns_the_first_selected_byte() {
        if !SimdBackend::Neon.is_available() {
            return;
        }
        let mut flags = [0u8; 16];
        flags[2] = 0b11111111;
        flags[11] = 0b11111111;
        // SAFETY: NEON is available and the input fills one vector.
        let actual = unsafe { scan(&flags, |bytes| selected_mask(bytes)) };
        assert_eq!(actual, 2);
    }

    #[test]
    fn rscan_returns_the_last_selected_byte() {
        if !SimdBackend::Neon.is_available() {
            return;
        }
        let mut flags = [0u8; 16];
        flags[2] = 0b11111111;
        flags[11] = 0b11111111;
        // SAFETY: NEON is available and the input fills one vector.
        let actual = unsafe { rscan(&flags, |bytes| selected_mask(bytes)) };
        assert_eq!(actual, 11);
    }
}
