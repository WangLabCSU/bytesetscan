#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// # Safety
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(in super::super) unsafe fn truffle_avx2<const REVERT: bool>(
    lower: &[u8; 16],
    upper: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // SAFETY: dispatch establishes CPU support; the scan handles short inputs.
    // Both table arrays provide one initialized 16-byte vector.
    unsafe {
        search::<REVERT>(
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

#[inline(always)]
unsafe fn search<const REVERT: bool>(lower: __m128i, upper: __m128i, haystack: &[u8]) -> usize {
    // SAFETY: the entry point establishes AVX2 support.
    // The short path is handled before full-width loads.
    // Aligned interior loads and the final overlapping vector stay in haystack.
    unsafe {
        let base = haystack.as_ptr();
        let len = haystack.len();
        // Copy a short slice into a zeroed vector, then exclude the padded
        // lanes from the result.
        if len < 32 {
            let mut bytes = [0u8; 32];
            bytes[..len].copy_from_slice(haystack);
            // Read 32 consecutive bytes with no alignment requirement. For a
            // short-input buffer the initialized padding is excluded after classification.
            let mut mask = mask32(lower, upper, _mm256_loadu_si256(bytes.as_ptr().cast()));
            if !REVERT {
                // Convert the classifier's nonmember bits into member bits. This
                // changes the predicate only; forward/reverse bit ordering is unchanged.
                mask = !mask;
            }
            // Discard positions outside the actual input, including any padding
            // whose zero byte happened to satisfy the selected predicate.
            mask &= u32::MAX.checked_shr((32 - len) as u32).unwrap_or(0);
            return if mask != 0 {
                // The lowest set bit identifies the first selected lane.
                mask.trailing_zeros() as usize
            } else {
                len
            };
        }

        // Read the 32 bytes starting here without requiring alignment.
        // Overlapping end loads still end exactly at the slice boundary.
        let mut mask = mask32(lower, upper, _mm256_loadu_si256(base.cast()));
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
        // Masking the low five address bits gives the remainder modulo 32.
        // 32 - remainder is the distance to the next 32-byte boundary
        // (32 bytes if already aligned). The initial block was checked,
        // so any overlap contains only rejected bytes.
        let mut start = 32 - (base as usize & 0b00011111);
        let last = len - 32;
        while start < last {
            // Read 32 bytes from the 32-byte-aligned interior address selected
            // by start/end; an aligned load still needs the full vector in bounds.
            let mut mask = mask32(lower, upper, _mm256_load_si256(base.add(start).cast()));
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
            start += 32;
        }
        // Read the 32 bytes starting here without requiring alignment.
        // Overlapping end loads still end exactly at the slice boundary.
        let mut mask = mask32(lower, upper, _mm256_loadu_si256(base.add(last).cast()));
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

// Reverse entry point. Requires avx2 any slice length.
#[target_feature(enable = "avx2")]
pub(in super::super) unsafe fn rtruffle_avx2<const REVERT: bool>(
    lower: &[u8; 16],
    upper: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // SAFETY: dispatch establishes CPU support; the scan handles short inputs.
    // Both table arrays provide one initialized 16-byte vector.
    unsafe {
        rsearch::<REVERT>(
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
unsafe fn rsearch<const REVERT: bool>(lower: __m128i, upper: __m128i, haystack: &[u8]) -> usize {
    // SAFETY: the entry point establishes AVX2 support.
    // The short path is handled before full-width loads.
    // Aligned interior loads and the final overlapping vector stay in haystack.
    unsafe {
        let base = haystack.as_ptr();
        let len = haystack.len();
        // Copy a short slice into a zeroed vector, then exclude the padded
        // lanes from the result.
        if len < 32 {
            let mut bytes = [0u8; 32];
            bytes[..len].copy_from_slice(haystack);
            // Read 32 consecutive bytes with no alignment requirement. For a
            // short-input buffer the initialized padding is excluded after classification.
            let mut mask = mask32(lower, upper, _mm256_loadu_si256(bytes.as_ptr().cast()));
            if !REVERT {
                // Convert the classifier's nonmember bits into member bits. This
                // changes the predicate only; forward/reverse bit ordering is unchanged.
                mask = !mask;
            }
            // Discard positions outside the actual input, including any padding
            // whose zero byte happened to satisfy the selected predicate.
            mask &= u32::MAX.checked_shr((32 - len) as u32).unwrap_or(0);
            return if mask != 0 {
                // 31 - leading_zeros locates the highest set bit in this u32,
                // which identifies the last selected byte lane.
                (31 - mask.leading_zeros()) as usize
            } else {
                len
            };
        }

        let last = len - 32;
        // Read the 32 bytes starting here without requiring alignment.
        // Overlapping end loads still end exactly at the slice boundary.
        let mut mask = mask32(lower, upper, _mm256_loadu_si256(base.add(last).cast()));
        if !REVERT {
            // Convert the classifier's nonmember bits into member bits. This
            // changes the predicate only; forward/reverse bit ordering is unchanged.
            mask = !mask;
        }
        if mask != 0 {
            // 31 - leading_zeros locates the highest set bit in this u32,
            // which identifies the last selected byte lane.
            // Add this block's start to return a position in the original input.
            return last + (31 - mask.leading_zeros()) as usize;
        }
        // An already aligned end may cause the initial suffix block to be
        // checked again.
        // The end address's low five bits give its remainder modulo 32.
        // Subtract that remainder from len to round base + end down to a
        // 32-byte boundary. The initial suffix scan covered the skipped bytes.
        let mut end = len - (base.add(len) as usize & 0b00011111);
        // The source compares buf_end with buf + 32 before decrementing.
        while end > 32 {
            end -= 32;
            // Read 32 bytes from the 32-byte-aligned interior address selected
            // by start/end; an aligned load still needs the full vector in bounds.
            let mut mask = mask32(lower, upper, _mm256_load_si256(base.add(end).cast()));
            if !REVERT {
                // Convert the classifier's nonmember bits into member bits. This
                // changes the predicate only; forward/reverse bit ordering is unchanged.
                mask = !mask;
            }
            if mask != 0 {
                // 31 - leading_zeros locates the highest set bit in this u32,
                // which identifies the last selected byte lane.
                // Add this block's start to return a position in the original input.
                return end + (31 - mask.leading_zeros()) as usize;
            }
        }
        // Read the 32 bytes starting here without requiring alignment.
        // Overlapping end loads still end exactly at the slice boundary.
        let mut mask = mask32(lower, upper, _mm256_loadu_si256(base.cast()));
        if !REVERT {
            // Convert the classifier's nonmember bits into member bits. This
            // changes the predicate only; forward/reverse bit ordering is unchanged.
            mask = !mask;
        }
        if mask != 0 {
            // 31 - leading_zeros locates the highest set bit in this u32,
            // which identifies the last selected byte lane.
            (31 - mask.leading_zeros()) as usize
        } else {
            len
        }
    }
}

#[inline(always)]
unsafe fn mask32(lower: __m128i, upper: __m128i, bytes: __m256i) -> u32 {
    // SAFETY: called only from an AVX2 entry point. pshufb zeros a lane
    // when bit 7 of the index is set; flipping that bit selects the half.
    unsafe {
        // set1_epi8(0b10000000u8 as i8) repeats the bit pattern 0b10000000 in every byte.
        // This is the shuffle index's zeroing bit, not a signed comparison.
        let top = _mm256_set1_epi8(0b1000_0000u8 as i8);
        // Repeating these eight byte values turns the high nibble into
        // one bit. For 0b11000001, the upper table supplies entry 1 and bit 4 is
        // tested; the lower-table shuffle is suppressed by the top bit.
        // Pack [1 << 0, ..., 1 << 7] in x86 little-endian byte order.
        // Broadcasting the packed u64 gives each 16-byte shuffle lane two
        // copies of the table; index h returns 1 << (h & 0b00000111).
        let bits = _mm256_set1_epi64x(u64::from_le_bytes([
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
        let lower = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(lower), bytes);
        // XOR with 0b10000000 toggles only the index's top bit, preserving the
        // low-nibble table index. This shuffle now contributes only for
        // bytes at least 128; exactly one of lower/upper can be nonzero.
        let upper = _mm256_shuffle_epi8(
            _mm256_broadcastsi128_si256(upper),
            _mm256_xor_si256(bytes, top),
        );
        // The 64-bit shift can carry bits across byte lanes. pshufb ignores
        // index bits 4..6, and clearing bit 7 prevents accidental zeroing;
        // the low nibble still contains this byte's original high nibble.
        // srli_epi64 shifts 64-bit lanes, not individual bytes. Its low
        // four output bits still equal this byte's high nibble; higher bits
        // may come from the next byte. andnot(top, shifted) clears bit 7
        // to prevent accidental PSHUFB zeroing; PSHUFB ignores bits 4..6.
        let index = _mm256_andnot_si256(top, _mm256_srli_epi64::<4>(bytes));
        // Shuffle the repeated powers of two by the high nibble. This
        // produces the membership bit 1 << ((byte >> 4) & 0b00000111) in every lane.
        // For 0b11000001 the upper table supplies entry 1 and selected is 0b00010000.
        let selected = _mm256_shuffle_epi8(bits, index);
        // OR combines the mutually exclusive lower/upper lookup results;
        // AND retains only the single membership bit chosen for this byte.
        let member = _mm256_and_si256(_mm256_or_si256(lower, upper), selected);
        // setzero supplies all-zero bytes. Equality with zero marks absent
        // membership bits as 0b11111111 and present bits as 0b00000000. movemask packs
        // each byte's top bit into the corresponding scalar position bit.
        _mm256_movemask_epi8(_mm256_cmpeq_epi8(member, _mm256_setzero_si256())) as u32
    }
}
