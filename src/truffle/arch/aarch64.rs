// Narrowing instruction reference:
// https://doc.rust-lang.org/core/arch/aarch64/fn.vshrn_n_u16.html

use core::arch::aarch64::*;

#[target_feature(enable = "neon")]
pub(super) unsafe fn search<const REVERT: bool>(
    lower: &[u8; 16],
    upper: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // SAFETY: dispatch establishes SIMD support. scan/rscan handle short inputs.
    // Table loads and every scan load below stay within initialized slices.
    unsafe {
        // Load all sixteen table entries into sixteen u8 lanes. No nibble
        // selection happens during this load; the following lookup uses them.
        let lower = vld1q_u8(lower.as_ptr());
        // Load all sixteen table entries into sixteen u8 lanes. No nibble
        // selection happens during this load; the following lookup uses them.
        let upper = vld1q_u8(upper.as_ptr());
        // Load two copies of [1 << 0, ..., 1 << 7]. The high nibble h
        // selects the membership bit 1 << (h & 0b00000111).
        let bits = vld1q_u8(
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
            .as_ptr(),
        );
        scan(haystack, |bytes| {
            // The outer AND tests a single membership bit in a table entry.
            // The table half, its entry, and that bit are selected separately below.
            let flags = vandq_u8(
                // vbsl(mask, yes, no) selects bits from yes where mask is 1 and
                // from no where it is 0. The comparison supplies whole-byte masks.
                vbslq_u8(
                    // Broadcast 0b10000000 and compare UNSIGNED bytes: lanes below 128 become
                    // 0b11111111, others 0b00000000. This selects lower for b < 128, upper
                    // otherwise.
                    vcltq_u8(bytes, vdupq_n_u8(0b10000000)),
                    // Broadcast 0b00001111 and AND with each byte; use b & 0b00001111 to look up
                    // the lower-half table entry. The selection mask may discard it.
                    // All indices here are in 0..16. The lookup's out-of-range
                    // zeroing behavior therefore does not participate in classification.
                    vqtbl1q_u8(lower, vandq_u8(bytes, vdupq_n_u8(0b00001111))),
                    // Use the same low-nibble index in the upper-half table. Unlike
                    // x86 PSHUFB, this lookup does not use bit 7 to choose the half.
                    // All indices here are in 0..16. The lookup's out-of-range
                    // zeroing behavior therefore does not participate in classification.
                    vqtbl1q_u8(upper, vandq_u8(bytes, vdupq_n_u8(0b00001111))),
                ),
                // Shift each unsigned byte right by 4, independently of its
                // neighbors. Lookup in bits converts h = b >> 4 into 0b00000001 << (h &
                // 0b00000111).
                // AND with the chosen table entry tests exactly that membership bit.
                // All indices here are in 0..16. The lookup's out-of-range
                // zeroing behavior therefore does not participate in classification.
                vqtbl1q_u8(bits, vshrq_n_u8::<4>(bytes)),
            );
            // Membership flags are zero for nonmembers and nonzero for members.
            // Each comparison produces 0b11111111 for a selected byte and
            // 0b00000000 otherwise, as required by selected_mask.
            let zero = vdupq_n_u8(0b00000000);
            let selected = if REVERT {
                vceqq_u8(flags, zero) // mismatch: select nonmembers.
            } else {
                vcgtq_u8(flags, zero) // find: select members; u8 comparison is unsigned.
            };
            selected_mask(selected)
        })
    }
}

#[inline(always)]
unsafe fn scan(haystack: &[u8], classify: impl Fn(uint8x16_t) -> u64) -> usize {
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
            let mask = classify(vld1q_u8(bytes.as_ptr()));
            // The first len bytes occupy the low bits on little-endian and
            // the high bits on big-endian. Clear every padding byte's flag.
            // A checked shift by 64 makes the valid mask zero for empty input.
            #[cfg(target_endian = "little")]
            let valid = u64::MAX.checked_shr((64 - len * 4) as u32).unwrap_or(0);
            #[cfg(target_endian = "big")]
            let valid = u64::MAX.checked_shl((64 - len * 4) as u32).unwrap_or(0);
            let mask = mask & valid;
            return if mask != 0 {
                // Locate the first selected byte using the mask's native bit order.
                // first_index accounts for the four bits per input byte.
                first_index(mask)
            } else {
                len
            };
        }
        // Read sixteen consecutive input bytes into a vector; this load
        // does not require a 16-byte-aligned address. The scan bounds or
        // initialized short-input buffer guarantee all sixteen bytes exist.
        let mask = classify(vld1q_u8(base));
        if mask != 0 {
            // Locate the first selected byte using the mask's native bit order.
            // first_index accounts for the four bits per input byte.
            return first_index(mask);
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
            let mask = classify(vld1q_u8(base.add(start)));
            if mask != 0 {
                // Locate the first selected byte using the mask's native bit order.
                // first_index accounts for the four bits per input byte.
                // Add this block's start to return a position in the original input.
                return start + first_index(mask);
            }
            start += 16;
        }
        // Read sixteen consecutive input bytes into a vector; this load
        // does not require a 16-byte-aligned address. The scan bounds or
        // initialized short-input buffer guarantee all sixteen bytes exist.
        let mask = classify(vld1q_u8(base.add(last)));
        if mask != 0 {
            // Locate the first selected byte using the mask's native bit order.
            // first_index accounts for the four bits per input byte.
            // Add this block's start to return a position in the original input.
            last + first_index(mask)
        } else {
            len
        }
    }
}

// Return the first selected byte's position in a nonzero mask.
#[inline(always)]
fn first_index(mask: u64) -> usize {
    // Input byte 0 occupies the lowest four bits on little-endian and the
    // highest four bits on big-endian. Count zeros from that end of the mask.
    #[cfg(target_endian = "little")]
    let zeros = mask.trailing_zeros();
    #[cfg(target_endian = "big")]
    let zeros = mask.leading_zeros();
    zeros as usize / 4
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn rsearch<const REVERT: bool>(
    lower: &[u8; 16],
    upper: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // SAFETY: dispatch establishes SIMD support. scan/rscan handle short inputs.
    // Table loads and every scan load below stay within initialized slices.
    unsafe {
        // Load all sixteen table entries into sixteen u8 lanes. No nibble
        // selection happens during this load; the following lookup uses them.
        let lower = vld1q_u8(lower.as_ptr());
        // Load all sixteen table entries into sixteen u8 lanes. No nibble
        // selection happens during this load; the following lookup uses them.
        let upper = vld1q_u8(upper.as_ptr());
        // Load two copies of [1 << 0, ..., 1 << 7]. The high nibble h
        // selects the membership bit 1 << (h & 0b00000111).
        let bits = vld1q_u8(
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
            .as_ptr(),
        );
        rscan(haystack, |bytes| {
            // The outer AND tests a single membership bit in a table entry.
            // The table half, its entry, and that bit are selected separately below.
            let flags = vandq_u8(
                // vbsl(mask, yes, no) selects bits from yes where mask is 1 and
                // from no where it is 0. The comparison supplies whole-byte masks.
                vbslq_u8(
                    // Broadcast 0b10000000 and compare UNSIGNED bytes: lanes below 128 become
                    // 0b11111111, others 0b00000000. This selects lower for b < 128, upper
                    // otherwise.
                    vcltq_u8(bytes, vdupq_n_u8(0b10000000)),
                    // Broadcast 0b00001111 and AND with each byte; use b & 0b00001111 to look up
                    // the lower-half table entry. The selection mask may discard it.
                    // All indices here are in 0..16. The lookup's out-of-range
                    // zeroing behavior therefore does not participate in classification.
                    vqtbl1q_u8(lower, vandq_u8(bytes, vdupq_n_u8(0b00001111))),
                    // Use the same low-nibble index in the upper-half table. Unlike
                    // x86 PSHUFB, this lookup does not use bit 7 to choose the half.
                    // All indices here are in 0..16. The lookup's out-of-range
                    // zeroing behavior therefore does not participate in classification.
                    vqtbl1q_u8(upper, vandq_u8(bytes, vdupq_n_u8(0b00001111))),
                ),
                // Shift each unsigned byte right by 4, independently of its
                // neighbors. Lookup in bits converts h = b >> 4 into 0b00000001 << (h &
                // 0b00000111).
                // AND with the chosen table entry tests exactly that membership bit.
                // All indices here are in 0..16. The lookup's out-of-range
                // zeroing behavior therefore does not participate in classification.
                vqtbl1q_u8(bits, vshrq_n_u8::<4>(bytes)),
            );
            // Membership flags are zero for nonmembers and nonzero for members.
            // Each comparison produces 0b11111111 for a selected byte and
            // 0b00000000 otherwise, as required by selected_mask.
            let zero = vdupq_n_u8(0b00000000);
            let selected = if REVERT {
                vceqq_u8(flags, zero) // mismatch: select nonmembers.
            } else {
                vcgtq_u8(flags, zero) // find: select members; u8 comparison is unsigned.
            };
            selected_mask(selected)
        })
    }
}

#[inline(always)]
unsafe fn rscan(haystack: &[u8], classify: impl Fn(uint8x16_t) -> u64) -> usize {
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
            let mask = classify(vld1q_u8(bytes.as_ptr()));
            // The first len bytes occupy the low bits on little-endian and
            // the high bits on big-endian. Clear every padding byte's flag.
            // A checked shift by 64 makes the valid mask zero for empty input.
            #[cfg(target_endian = "little")]
            let valid = u64::MAX.checked_shr((64 - len * 4) as u32).unwrap_or(0);
            #[cfg(target_endian = "big")]
            let valid = u64::MAX.checked_shl((64 - len * 4) as u32).unwrap_or(0);
            let mask = mask & valid;
            return if mask != 0 {
                // Locate the last selected byte using the mask's native bit order.
                // last_index accounts for the four bits per input byte.
                last_index(mask)
            } else {
                len
            };
        }
        let last = len - 16;
        // Read sixteen consecutive input bytes into a vector; this load
        // does not require a 16-byte-aligned address. The scan bounds or
        // initialized short-input buffer guarantee all sixteen bytes exist.
        let mask = classify(vld1q_u8(base.add(last)));
        if mask != 0 {
            // Locate the last selected byte using the mask's native bit order.
            // last_index accounts for the four bits per input byte.
            // Add this block's start to return a position in the original input.
            return last + last_index(mask);
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
            let mask = classify(vld1q_u8(base.add(end)));
            if mask != 0 {
                // Locate the last selected byte using the mask's native bit order.
                // last_index accounts for the four bits per input byte.
                // Add this block's start to return a position in the original input.
                return end + last_index(mask);
            }
        }
        // Read sixteen consecutive input bytes into a vector; this load
        // does not require a 16-byte-aligned address. The scan bounds or
        // initialized short-input buffer guarantee all sixteen bytes exist.
        let mask = classify(vld1q_u8(base));
        if mask != 0 {
            // Locate the last selected byte using the mask's native bit order.
            // last_index accounts for the four bits per input byte.
            last_index(mask)
        } else {
            len
        }
    }
}

// Return the last selected byte's position in a nonzero mask.
#[inline(always)]
fn last_index(mask: u64) -> usize {
    // Input byte 15 occupies the highest four bits on little-endian and the
    // lowest four bits on big-endian. Count from that end, then convert the
    // distance back to a position measured from input byte 0.
    #[cfg(target_endian = "little")]
    let zeros = mask.leading_zeros();
    #[cfg(target_endian = "big")]
    let zeros = mask.trailing_zeros();
    (63 - zeros as usize) / 4
}

// Pack 0b00000000/0b11111111 byte flags into four bits per input byte.
// Keep native bit order: byte i starts at bit 4*i on little-endian and
// bit 4*(15-i) on big-endian. The scanners choose the corresponding end
// when locating the first or last selected byte.
#[inline(always)]
unsafe fn selected_mask(selected: uint8x16_t) -> u64 {
    // SAFETY: the caller establishes NEON support. These operations only
    // reinterpret, narrow, and extract values already held in registers.
    unsafe {
        // Read each pair of flags as a u16, shift right by four, then keep
        // its low byte. Each original flag contributes one 0b0000/0b1111
        // nibble. On little-endian, [0b11111111, 0] becomes 0b00001111;
        // on big-endian, the same pair becomes 0b11110000.
        let packed = vshrn_n_u16::<4>(vreinterpretq_u16_u8(selected));
        vget_lane_u64::<0>(vreinterpret_u64_u8(packed))
    }
}

#[cfg(test)]
mod tests {
    use super::{rscan, scan, selected_mask, vld1q_u8};
    use crate::SimdBackend;

    #[test]
    fn selected_mask_uses_native_byte_order() {
        if !SimdBackend::Neon.is_available() {
            return;
        }
        for position in 0..16 {
            let mut flags = [0u8; 16];
            flags[position] = 0b11111111;
            // SAFETY: NEON is available and all sixteen flags are initialized.
            let mask = unsafe { selected_mask(vld1q_u8(flags.as_ptr())) };
            #[cfg(target_endian = "little")]
            let expected = 0b1111u64 << (position * 4);
            #[cfg(target_endian = "big")]
            let expected = 0b1111u64 << ((15 - position) * 4);
            assert_eq!(mask, expected, "byte {position}");
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
    fn scan_returns_the_first_selected_byte_in_a_later_block() {
        if !SimdBackend::Neon.is_available() {
            return;
        }
        let mut flags = [0u8; 64];
        flags[35] = 0b11111111;
        flags[55] = 0b11111111;
        // SAFETY: NEON is available and scan bounds all vector loads.
        let actual = unsafe { scan(&flags, |bytes| selected_mask(bytes)) };
        assert_eq!(actual, 35);
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

    #[test]
    fn rscan_returns_the_last_selected_byte_in_an_earlier_block() {
        if !SimdBackend::Neon.is_available() {
            return;
        }
        let mut flags = [0u8; 64];
        flags[5] = 0b11111111;
        flags[25] = 0b11111111;
        // SAFETY: NEON is available and rscan bounds all vector loads.
        let actual = unsafe { rscan(&flags, |bytes| selected_mask(bytes)) };
        assert_eq!(actual, 25);
    }

    #[test]
    fn scan_preserves_short_input_positions() {
        if !SimdBackend::Neon.is_available() {
            return;
        }
        for len in 1..16 {
            for position in 0..len {
                let mut flags = [0u8; 16];
                flags[position] = 0b11111111;
                // SAFETY: NEON is available; scan handles the partial vector.
                let actual = unsafe { scan(&flags[..len], |bytes| selected_mask(bytes)) };
                assert_eq!(actual, position, "len={len}, byte={position}");
            }
        }
    }

    #[test]
    fn rscan_preserves_short_input_positions() {
        if !SimdBackend::Neon.is_available() {
            return;
        }
        for len in 1..16 {
            for position in 0..len {
                let mut flags = [0u8; 16];
                flags[position] = 0b11111111;
                // SAFETY: NEON is available; rscan handles the partial vector.
                let actual = unsafe { rscan(&flags[..len], |bytes| selected_mask(bytes)) };
                assert_eq!(actual, position, "len={len}, byte={position}");
            }
        }
    }

    #[test]
    fn scan_excludes_matching_padding() {
        use super::{vceqq_u8, vdupq_n_u8};

        if !SimdBackend::Neon.is_available() {
            return;
        }
        // The input never matches zero, but every zero-filled padding byte does.
        for len in 0..16 {
            let bytes = [1u8; 16];
            // SAFETY: NEON is available; scan handles the partial vector.
            let actual = unsafe {
                scan(&bytes[..len], |bytes| {
                    selected_mask(vceqq_u8(bytes, vdupq_n_u8(0)))
                })
            };
            assert_eq!(actual, len);
        }
    }

    #[test]
    fn rscan_excludes_matching_padding() {
        use super::{vceqq_u8, vdupq_n_u8};

        if !SimdBackend::Neon.is_available() {
            return;
        }
        // The input never matches zero, but every zero-filled padding byte does.
        for len in 0..16 {
            let bytes = [1u8; 16];
            // SAFETY: NEON is available; rscan handles the partial vector.
            let actual = unsafe {
                rscan(&bytes[..len], |bytes| {
                    selected_mask(vceqq_u8(bytes, vdupq_n_u8(0)))
                })
            };
            assert_eq!(actual, len);
        }
    }
}
