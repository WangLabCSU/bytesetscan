#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// Requires ssse3, valid table arrays, any slice length.
#[target_feature(enable = "ssse3")]
pub(in super::super) fn truffle_ssse3<const REVERT: bool>(
    lower: &[u8; 16],
    upper: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // SAFETY: dispatch establishes CPU support; the scan handles short inputs.
    // Both table arrays provide one initialized 16-byte vector.
    unsafe {
        search128::<REVERT>(
            // Load the 16 initialized table entries into one register; the
            // array has byte alignment, so use the unaligned load intrinsic.
            _mm_loadu_si128(lower.as_ptr().cast()),
            // Load the 16 initialized table entries into one register; the
            // array has byte alignment, so use the unaligned load intrinsic.
            _mm_loadu_si128(upper.as_ptr().cast()),
            haystack,
        )
    }
}

// SSSE3 and SSE4.2 share the initial/aligned/final 16-byte scan.
#[inline(always)]
pub(super) unsafe fn search128<const REVERT: bool>(
    lower: __m128i,
    upper: __m128i,
    haystack: &[u8],
) -> usize {
    // SAFETY: the entry point establishes SSSE3 support.
    // The short path is handled before full-width loads.
    // Aligned interior loads and the final overlapping vector stay in haystack.
    unsafe {
        let base = haystack.as_ptr();
        let len = haystack.len();
        // Copy a short slice into a zeroed vector, then exclude the padded
        // lanes from the result.
        if len < 16 {
            let mut bytes = [0u8; 16];
            bytes[..len].copy_from_slice(haystack);
            // Load sixteen consecutive bytes without assuming 16-byte alignment.
            // The short path or scan bounds guarantee all sixteen are readable.
            let mut mask = mask16(lower, upper, _mm_loadu_si128(bytes.as_ptr().cast()));
            if !REVERT {
                // Convert the classifier's nonmember bits into member bits. This
                // changes the predicate only; forward/reverse bit ordering is unchanged.
                mask = !mask;
            }
            // Discard positions outside the actual input, including any padding
            // whose zero byte happened to satisfy the selected predicate.
            mask &= u16::MAX.checked_shr((16 - len) as u32).unwrap_or(0);
            return if mask != 0 {
                // The lowest set bit identifies the first selected lane.
                mask.trailing_zeros() as usize
            } else {
                len
            };
        }
        // Load sixteen consecutive bytes without assuming 16-byte alignment.
        // The short path or scan bounds guarantee all sixteen are readable.
        let mut mask = mask16(lower, upper, _mm_loadu_si128(base.cast()));
        if !REVERT {
            // Convert the classifier's nonmember bits into member bits. This
            // changes the predicate only; forward/reverse bit ordering is unchanged.
            mask = !mask;
        }
        if mask != 0 {
            // The lowest set bit identifies the first selected lane.
            return mask.trailing_zeros() as usize;
        }
        // After the first unaligned block, advance to the next boundary.
        // Each iteration classifies one vector.
        // Masking the low four address bits gives the remainder modulo 16.
        // 16 - remainder is the distance to the next 16-byte boundary
        // (16 bytes if already aligned). The initial block was checked,
        // so any overlap contains only rejected bytes.
        let mut start = 16 - (base as usize & 0b00001111);
        let last = len - 16;
        while start < last {
            // Load sixteen bytes from the 16-byte-aligned interior address;
            // the loop excludes positions without a complete input vector.
            let mut mask = mask16(lower, upper, _mm_load_si128(base.add(start).cast()));
            if !REVERT {
                // Convert the classifier's nonmember bits into member bits. This
                // changes the predicate only; forward/reverse bit ordering is unchanged.
                mask = !mask;
            }
            if mask != 0 {
                // The lowest set bit identifies the first selected lane.
                // Add this block's start to return a position in the original input.
                return start + mask.trailing_zeros() as usize;
            }
            start += 16;
        }
        // Load sixteen consecutive bytes without assuming 16-byte alignment.
        // The short path or scan bounds guarantee all sixteen are readable.
        let mut mask = mask16(lower, upper, _mm_loadu_si128(base.add(last).cast()));
        if !REVERT {
            // Convert the classifier's nonmember bits into member bits. This
            // changes the predicate only; forward/reverse bit ordering is unchanged.
            mask = !mask;
        }
        if mask != 0 {
            // The lowest set bit identifies the first selected lane.
            // Add this block's start to return a position in the original input.
            last + mask.trailing_zeros() as usize
        } else {
            len
        }
    }
}

// Reverse entry point. Requires ssse3 any slice length.
#[target_feature(enable = "ssse3")]
pub(in super::super) fn rtruffle_ssse3<const REVERT: bool>(
    lower: &[u8; 16],
    upper: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // SAFETY: dispatch establishes CPU support; the scan handles short inputs.
    // Both table arrays provide one initialized 16-byte vector.
    unsafe {
        rsearch128::<REVERT>(
            // Load the 16 initialized table entries into one register; the
            // array has byte alignment, so use the unaligned load intrinsic.
            _mm_loadu_si128(lower.as_ptr().cast()),
            // Load the 16 initialized table entries into one register; the
            // array has byte alignment, so use the unaligned load intrinsic.
            _mm_loadu_si128(upper.as_ptr().cast()),
            haystack,
        )
    }
}

// Reverse counterpart to the forward scan: unaligned suffix, aligned interior,
// then an overlapping prefix. The highest set bit identifies the last match.
#[inline(always)]
pub(super) unsafe fn rsearch128<const REVERT: bool>(
    lower: __m128i,
    upper: __m128i,
    haystack: &[u8],
) -> usize {
    // SAFETY: the entry point establishes SSSE3 support.
    // The short path is handled before full-width loads.
    // Aligned interior loads and the final overlapping vector stay in haystack.
    unsafe {
        let base = haystack.as_ptr();
        let len = haystack.len();
        // Copy a short slice into a zeroed vector, then exclude the padded
        // lanes from the result.
        if len < 16 {
            let mut bytes = [0u8; 16];
            bytes[..len].copy_from_slice(haystack);
            // Load sixteen consecutive bytes without assuming 16-byte alignment.
            // The short path or scan bounds guarantee all sixteen are readable.
            let mut mask = mask16(lower, upper, _mm_loadu_si128(bytes.as_ptr().cast()));
            if !REVERT {
                // Convert the classifier's nonmember bits into member bits. This
                // changes the predicate only; forward/reverse bit ordering is unchanged.
                mask = !mask;
            }
            // Discard positions outside the actual input, including any padding
            // whose zero byte happened to satisfy the selected predicate.
            mask &= u16::MAX.checked_shr((16 - len) as u32).unwrap_or(0);
            return if mask != 0 {
                // 15 - leading_zeros locates the highest set bit in this u16,
                // which identifies the last selected byte lane.
                (15 - mask.leading_zeros()) as usize
            } else {
                len
            };
        }
        let last = len - 16;
        // Load sixteen consecutive bytes without assuming 16-byte alignment.
        // The short path or scan bounds guarantee all sixteen are readable.
        let mut mask = mask16(lower, upper, _mm_loadu_si128(base.add(last).cast()));
        if !REVERT {
            // Convert the classifier's nonmember bits into member bits. This
            // changes the predicate only; forward/reverse bit ordering is unchanged.
            mask = !mask;
        }
        if mask != 0 {
            return last + (31 - (mask as u32).leading_zeros()) as usize;
        }
        // An already aligned end may cause the initial suffix block to be
        // checked again.
        // The end address's low four bits give its remainder modulo 16.
        // Subtract that remainder from len to round base + end down to a
        // 16-byte boundary. The initial suffix scan covered the skipped bytes.
        let mut end = len - (base.add(len) as usize & 0b00001111);
        // The source compares buf_end with buf + 16 before decrementing.
        while end > 16 {
            end -= 16;
            // Load sixteen bytes from the 16-byte-aligned interior address;
            // the loop excludes positions without a complete input vector.
            let mut mask = mask16(lower, upper, _mm_load_si128(base.add(end).cast()));
            if !REVERT {
                // Convert the classifier's nonmember bits into member bits. This
                // changes the predicate only; forward/reverse bit ordering is unchanged.
                mask = !mask;
            }
            if mask != 0 {
                return end + (31 - (mask as u32).leading_zeros()) as usize;
            }
        }
        // Load sixteen consecutive bytes without assuming 16-byte alignment.
        // The short path or scan bounds guarantee all sixteen are readable.
        let mut mask = mask16(lower, upper, _mm_loadu_si128(base.cast()));
        if !REVERT {
            // Convert the classifier's nonmember bits into member bits. This
            // changes the predicate only; forward/reverse bit ordering is unchanged.
            mask = !mask;
        }
        if mask != 0 {
            (31 - (mask as u32).leading_zeros()) as usize
        } else {
            len
        }
    }
}

#[inline(always)]
unsafe fn mask16(lower: __m128i, upper: __m128i, bytes: __m128i) -> u16 {
    // SAFETY: the caller has established SSSE3 support. The SSSE3 and
    // SSE4.2 scans share this classifier for a single 16-byte block.
    unsafe {
        // set1_epi8(0b10000000u8 as i8) repeats the bit pattern 0b10000000 in every byte.
        // This is the shuffle index's zeroing bit, not a signed comparison.
        let top = _mm_set1_epi8(0b10000000u8 as i8);
        // Pack [1 << 0, ..., 1 << 7] in x86 little-endian byte order.
        // Broadcasting the packed u64 gives each 16-byte shuffle lane two
        // copies of the table; index h returns 1 << (h & 0b00000111).
        let bits = _mm_set1_epi64x(u64::from_le_bytes([
            1 << 0,
            1 << 1,
            1 << 2,
            1 << 3,
            1 << 4,
            1 << 5,
            1 << 6,
            1 << 7,
        ]) as i64);
        // PSHUFB selects table[byte & 0b00001111] unless index bit 7 is set,
        // in which case it returns zero. Thus this lookup contributes only
        // for bytes below 128. On wide vectors, broadcast repeats the full
        // 16-byte table in each independent 128-bit shuffle lane.
        let lower = _mm_shuffle_epi8(lower, bytes);
        // XOR with 0b10000000 toggles only the index's top bit, preserving the
        // low-nibble table index. This shuffle now contributes only for
        // bytes at least 128; exactly one of lower/upper can be nonzero.
        let upper = _mm_shuffle_epi8(upper, _mm_xor_si128(bytes, top));
        // srli_epi64 shifts 64-bit lanes, not individual bytes. Its low
        // four output bits still equal this byte's high nibble; higher bits
        // may come from the next byte. andnot(top, shifted) clears bit 7
        // to prevent accidental PSHUFB zeroing; PSHUFB ignores bits 4..6.
        let index = _mm_andnot_si128(top, _mm_srli_epi64::<4>(bytes));
        // Shuffle the repeated powers of two by the high nibble. This
        // produces the membership bit 1 << ((byte >> 4) & 0b00000111) in every lane.
        // For 0b11000001 the upper table supplies entry 1 and selected is 0b00010000.
        let selected = _mm_shuffle_epi8(bits, index);
        // OR combines the mutually exclusive lower/upper lookup results;
        // AND retains only the single membership bit chosen for this byte.
        let member = _mm_and_si128(_mm_or_si128(lower, upper), selected);
        // setzero supplies all-zero bytes. Equality with zero marks absent
        // membership bits as 0b11111111 and present bits as 0b00000000. movemask packs
        // each byte's top bit into the corresponding scalar position bit.
        _mm_movemask_epi8(_mm_cmpeq_epi8(member, _mm_setzero_si128())) as u16
    }
}
