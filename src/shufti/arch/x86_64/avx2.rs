#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// # Safety
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(in super::super) fn shufti_avx2<const REVERT: bool>(
    low: &[u8; 16],
    high: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // Use scalar matching when the input cannot fill a 16-byte vector.
    if haystack.len() < 16 {
        return super::super::search_fallback::<REVERT>(low, high, haystack);
    }
    // SAFETY: each array supplies exactly one initialized 16-byte vector;
    // dispatch establishes CPU support and the short-input check above
    // establishes the minimum length needed by this scan.
    unsafe {
        search::<REVERT>(
            // Load the 16 initialized table entries into one register; the
            // array has byte alignment, so use the unaligned load intrinsic.
            _mm_loadu_si128(low.as_ptr().cast()),
            // Load the 16 initialized table entries into one register; the
            // array has byte alignment, so use the unaligned load intrinsic.
            _mm_loadu_si128(high.as_ptr().cast()),
            haystack,
        )
    }
}

#[inline(always)]
unsafe fn search<const REVERT: bool>(low: __m128i, high: __m128i, haystack: &[u8]) -> usize {
    // SAFETY: the entry point establishes AVX2 support and len >= 16.
    // The short path uses two in-bounds 16-byte loads.
    // Aligned interior loads and the final overlapping vector stay in haystack.
    unsafe {
        let ptr = haystack.as_ptr();
        let len = haystack.len();
        // shuftiFwdShort / shuftiRevShort use two overlapping 16-byte reads.
        if len <= 32 {
            // Load sixteen consecutive bytes without assuming 16-byte alignment.
            // The short path or scan bounds guarantee all sixteen are readable.
            let mut mask = mask16_avx2(low, high, _mm_loadu_si128(ptr.cast()));
            if !REVERT {
                // Convert the classifier's nonmember bits into member bits. This
                // changes the predicate only; forward/reverse bit ordering is unchanged.
                mask = !mask;
            }
            if mask != 0 {
                // The lowest set bit identifies the first selected lane.
                return mask.trailing_zeros() as usize;
            }
            let last = len - 16;
            // Load sixteen consecutive bytes without assuming 16-byte alignment.
            // The short path or scan bounds guarantee all sixteen are readable.
            let mut mask = mask16_avx2(low, high, _mm_loadu_si128(ptr.add(last).cast()));
            if !REVERT {
                // Convert the classifier's nonmember bits into member bits. This
                // changes the predicate only; forward/reverse bit ordering is unchanged.
                mask = !mask;
            }
            return if mask != 0 {
                // The lowest set bit identifies the first selected lane.
                // Add this block's start to return a position in the original input.
                last + mask.trailing_zeros() as usize
            } else {
                len
            };
        }
        // Read the 32 bytes starting here without requiring alignment.
        // Overlapping end loads still end exactly at the slice boundary.
        let mut mask = mask32(low, high, _mm256_loadu_si256(ptr.cast()));
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
        let mut start = 32 - (ptr as usize & 0b00011111);
        let last = len - 32;
        while start < last {
            // Read 32 bytes from the 32-byte-aligned interior address selected
            // by start/end; an aligned load still needs the full vector in bounds.
            let mut mask = mask32(low, high, _mm256_load_si256(ptr.add(start).cast()));
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
        let mut mask = mask32(low, high, _mm256_loadu_si256(ptr.add(last).cast()));
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
pub(in super::super) fn rshufti_avx2<const REVERT: bool>(
    low: &[u8; 16],
    high: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // Use scalar matching when the input cannot fill a 16-byte vector.
    if haystack.len() < 16 {
        return super::super::rsearch_fallback::<REVERT>(low, high, haystack);
    }
    // SAFETY: each array supplies exactly one initialized 16-byte vector;
    // dispatch establishes CPU support and the short-input check above
    // establishes the minimum length needed by this scan.
    unsafe {
        rsearch::<REVERT>(
            // Load the 16 initialized table entries into one register; the
            // array has byte alignment, so use the unaligned load intrinsic.
            _mm_loadu_si128(low.as_ptr().cast()),
            // Load the 16 initialized table entries into one register; the
            // array has byte alignment, so use the unaligned load intrinsic.
            _mm_loadu_si128(high.as_ptr().cast()),
            haystack,
        )
    }
}

// Reverse counterpart to the forward scan: unaligned suffix, aligned interior,
// then an overlapping prefix. The highest set bit identifies the last match.
#[inline(always)]
unsafe fn rsearch<const REVERT: bool>(low: __m128i, high: __m128i, haystack: &[u8]) -> usize {
    // SAFETY: the entry point establishes AVX2 support and len >= 16.
    // The short path uses two in-bounds 16-byte loads.
    // Aligned interior loads and the final overlapping vector stay in haystack.
    unsafe {
        let ptr = haystack.as_ptr();
        let len = haystack.len();
        // shuftiFwdShort / shuftiRevShort use two overlapping 16-byte reads.
        if len <= 32 {
            let last = len - 16;
            // Load sixteen consecutive bytes without assuming 16-byte alignment.
            // The short path or scan bounds guarantee all sixteen are readable.
            let mut mask = mask16_avx2(low, high, _mm_loadu_si128(ptr.add(last).cast()));
            if !REVERT {
                // Convert the classifier's nonmember bits into member bits. This
                // changes the predicate only; forward/reverse bit ordering is unchanged.
                mask = !mask;
            }
            if mask != 0 {
                return last + (31 - (mask as u32).leading_zeros()) as usize;
            }
            // Load sixteen consecutive bytes without assuming 16-byte alignment.
            // The short path or scan bounds guarantee all sixteen are readable.
            let mut mask = mask16_avx2(low, high, _mm_loadu_si128(ptr.cast()));
            if !REVERT {
                // Convert the classifier's nonmember bits into member bits. This
                // changes the predicate only; forward/reverse bit ordering is unchanged.
                mask = !mask;
            }
            return if mask != 0 {
                (31 - (mask as u32).leading_zeros()) as usize
            } else {
                len
            };
        }
        let last = len - 32;
        // Read the 32 bytes starting here without requiring alignment.
        // Overlapping end loads still end exactly at the slice boundary.
        let mut mask = mask32(low, high, _mm256_loadu_si256(ptr.add(last).cast()));
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
        // Subtract that remainder from len to round ptr + end down to a
        // 32-byte boundary. The initial suffix scan covered the skipped bytes.
        let mut end = len - (ptr.add(len) as usize & 0b00011111);
        // The source compares buf_end with buf + 32 before decrementing.
        while end > 32 {
            end -= 32;
            // Read 32 bytes from the 32-byte-aligned interior address selected
            // by start/end; an aligned load still needs the full vector in bounds.
            let mut mask = mask32(low, high, _mm256_load_si256(ptr.add(end).cast()));
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
        let mut mask = mask32(low, high, _mm256_loadu_si256(ptr.cast()));
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
unsafe fn mask32(low: __m128i, high: __m128i, bytes: __m256i) -> u32 {
    // SAFETY: called only from an AVX2 entry point.
    unsafe {
        // set1_epi8 broadcasts 0b00001111 into every byte lane. AND with this
        // mask extracts each input byte's low nibble as a table index in 0..16.
        let nibble = _mm256_set1_epi8(0b00001111);
        // shuffle_epi8 performs the low-table lookup independently in each
        // 128-bit lane. The following broadcast supplies an identical table
        // to every lane; AND supplies the input bytes' low-nibble indices.
        let low = _mm256_shuffle_epi8(
            // Replicate the complete 16-byte low table across the wide register;
            // this does not merge the shuffle's separate 128-bit lookup domains.
            _mm256_broadcastsi128_si256(low),
            _mm256_and_si256(bytes, nibble),
        );
        // Look up the high nibble in the high table. andnot(a, b) means
        // (~a) & b: clear each byte's low four bits BEFORE shifting the
        // 64-bit lanes. This prevents a neighboring byte's low nibble from
        // spilling into this byte; the resulting indices are exactly b >> 4.
        let high = _mm256_shuffle_epi8(
            // Repeat the 16-byte high table in each 128-bit shuffle lane.
            _mm256_broadcastsi128_si256(high),
            _mm256_srli_epi64::<4>(_mm256_andnot_si256(nibble, bytes)),
        );
        // AND intersects the two sets of allowed groups. cmpeq_epi8 with
        // a zero vector yields 0b11111111 for a nonmember and 0b00000000 for a member.
        // movemask_epi8 collects each byte's top bit into scalar bit i;
        // the cast preserves the meaningful byte-position bits, including the top one.
        _mm256_movemask_epi8(_mm256_cmpeq_epi8(
            _mm256_and_si256(low, high),
            _mm256_setzero_si256(),
        )) as u32
    }
}

#[inline(always)]
unsafe fn mask16_avx2(low: __m128i, high: __m128i, bytes: __m128i) -> u16 {
    // SAFETY: called only from an AVX2 entry point. Pack both nibble
    // lookups into one AVX2 shuffle.
    unsafe {
        // set_m128i takes (high_half, low_half): lanes 0..16 hold the low
        // table, lanes 16..32 the high table. Two different lookups can now
        // run in one AVX2 shuffle, still respecting its 128-bit lane boundary.
        let table = _mm256_set_m128i(high, low);
        // Duplicate the same sixteen input bytes into the low half and
        // their 64-bit-lane-shifted version into the high half. The later
        // 0b00001111 AND discards cross-byte spill bits, yielding low/high indices.
        let index = _mm256_set_m128i(_mm_srli_epi64::<4>(bytes), bytes);
        // set1_epi8(0b00001111) fills all 32 index masks; AND isolates each nibble.
        // PSHUFB returns low-group flags in the lower 128 bits and high-group
        // flags for those SAME input bytes in the upper 128 bits.
        let flags =
            _mm256_shuffle_epi8(table, _mm256_and_si256(index, _mm256_set1_epi8(0b00001111)));
        // Combine flags for corresponding input bytes, not adjacent bytes.
        let member = _mm_and_si128(
            // View the low 128 bits as sixteen bytes; this cast does not
            // numerically convert or copy the lanes. These are the low flags.
            _mm256_castsi256_si128(flags),
            // Extract 128-bit lane 1, the high flags; AND with lane 0 leaves
            // a nonzero byte precisely when some group accepts both nibbles.
            _mm256_extracti128_si256::<1>(flags),
        );
        // AND intersects the two sets of allowed groups. cmpeq_epi8 with
        // a zero vector yields 0b11111111 for a nonmember and 0b00000000 for a member.
        // movemask_epi8 collects each byte's top bit into scalar bit i;
        // the cast preserves the meaningful byte-position bits, including the top one.
        _mm_movemask_epi8(_mm_cmpeq_epi8(member, _mm_setzero_si128())) as u16
    }
}
