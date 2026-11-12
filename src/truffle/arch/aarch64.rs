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
        // Load two repetitions of [0b00000001, 0b00000010, 0b00000100, 0b00001000, 0b00010000,
        // 0b00100000, 0b01000000, 0b10000000]. The
        // high nibble h will select the bit 0b00000001 << (h & 0b00000111) from this vector.
        let bits = vld1q_u8(
            [
                0b00000001u8,
                0b00000010,
                0b00000100,
                0b00001000,
                0b00010000,
                0b00100000,
                0b01000000,
                0b10000000,
                0b00000001,
                0b00000010,
                0b00000100,
                0b00001000,
                0b00010000,
                0b00100000,
                0b01000000,
                0b10000000,
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
            // Broadcast zero and compare each u8 lane: no membership flag
            // becomes 0b11111111, while a member becomes 0b00000000. These are nonmember flags.
            let mut selected = vceqq_u8(flags, vdupq_n_u8(0b00000000));
            if !REVERT {
                // Bitwise NOT switches nonmember flags into member flags for find;
                // mismatch leaves the comparison result unchanged.
                selected = vmvnq_u8(selected);
            }
            // Each input byte contributes four identical bits in lane order.
            // reinterpret changes only the lane view: sixteen 0b00000000/0b11111111 bytes
            // become eight u16 lanes. SHRN shifts each u16 right by 4 and keeps
            // its low eight bits, merging a pair of byte flags into two nibbles.
            // For example, flags [0b11111111, 0b00000000] yield 0b00001111; [0b00000000,
            // 0b11111111] yield
            // 0b11110000. Each original byte is now represented by four identical bits.
            let packed = vshrn_n_u16::<4>(vreinterpretq_u16_u8(selected));
            // View the eight packed bytes as one u64 and extract lane 0.
            // The scan divides the selected bit position by 4 to recover the
            // original byte lane, in both forward and reverse directions.
            vget_lane_u64::<0>(vreinterpret_u64_u8(packed))
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
            let mask = classify(vld1q_u8(bytes.as_ptr()))
                & u64::MAX.checked_shr((64 - len * 4) as u32).unwrap_or(0);
            return if mask != 0 {
                // The lowest set bit identifies the first selected lane.
                // Each lane occupies four mask bits, so divide that bit index by 4.
                mask.trailing_zeros() as usize / 4
            } else {
                len
            };
        }
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
        // Load two repetitions of [0b00000001, 0b00000010, 0b00000100, 0b00001000, 0b00010000,
        // 0b00100000, 0b01000000, 0b10000000]. The
        // high nibble h will select the bit 0b00000001 << (h & 0b00000111) from this vector.
        let bits = vld1q_u8(
            [
                0b00000001u8,
                0b00000010,
                0b00000100,
                0b00001000,
                0b00010000,
                0b00100000,
                0b01000000,
                0b10000000,
                0b00000001,
                0b00000010,
                0b00000100,
                0b00001000,
                0b00010000,
                0b00100000,
                0b01000000,
                0b10000000,
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
            // Broadcast zero and compare each u8 lane: no membership flag
            // becomes 0b11111111, while a member becomes 0b00000000. These are nonmember flags.
            let mut selected = vceqq_u8(flags, vdupq_n_u8(0b00000000));
            if !REVERT {
                // Bitwise NOT switches nonmember flags into member flags for find;
                // mismatch leaves the comparison result unchanged.
                selected = vmvnq_u8(selected);
            }
            // Each input byte contributes four identical bits in lane order.
            // reinterpret changes only the lane view: sixteen 0b00000000/0b11111111 bytes
            // become eight u16 lanes. SHRN shifts each u16 right by 4 and keeps
            // its low eight bits, merging a pair of byte flags into two nibbles.
            // For example, flags [0b11111111, 0b00000000] yield 0b00001111; [0b00000000,
            // 0b11111111] yield
            // 0b11110000. Each original byte is now represented by four identical bits.
            let packed = vshrn_n_u16::<4>(vreinterpretq_u16_u8(selected));
            // View the eight packed bytes as one u64 and extract lane 0.
            // The scan divides the selected bit position by 4 to recover the
            // original byte lane, in both forward and reverse directions.
            vget_lane_u64::<0>(vreinterpret_u64_u8(packed))
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
            let mask = classify(vld1q_u8(bytes.as_ptr()))
                & u64::MAX.checked_shr((64 - len * 4) as u32).unwrap_or(0);
            return if mask != 0 {
                // 63 - leading_zeros locates the highest set bit in this u64.
                // Divide by 4 to convert its four-bit nibble back to a byte lane.
                (63 - mask.leading_zeros() as usize) / 4
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
