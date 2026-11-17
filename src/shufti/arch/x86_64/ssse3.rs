#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// Requires ssse3, valid table arrays, any slice length.
#[target_feature(enable = "ssse3")]
pub(in super::super) fn shufti_ssse3<const REVERT: bool>(
    low: &[u8; 16],
    high: &[u8; 16],
    haystack: &[u8],
) -> usize {
    if haystack.len() < 8 {
        return super::super::search_fallback::<REVERT>(low, high, haystack);
    }
    if haystack.len() < 16 {
        // SAFETY: SSSE3 is enabled, the table arrays contain sixteen bytes,
        // and the checks above establish 8 <= haystack.len() < 16.
        return unsafe {
            search_short::<REVERT>(
                _mm_loadu_si128(low.as_ptr().cast()),
                _mm_loadu_si128(high.as_ptr().cast()),
                haystack,
            )
        };
    }
    // SAFETY: array loads stay within their initialized storage; dispatch
    // validates CPU support; short inputs are handled before full input loads.
    unsafe {
        search128::<REVERT>(
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

// SSSE3 and SSE4.2 share the initial/aligned/final 16-byte scan.
#[inline(always)]
unsafe fn search128<const REVERT: bool>(low: __m128i, high: __m128i, haystack: &[u8]) -> usize {
    // SAFETY: the entry point establishes SSSE3 support and len >= 16.
    // Aligned interior loads and the final overlapping vector stay in haystack.
    unsafe {
        let ptr = haystack.as_ptr();
        let len = haystack.len();
        // Load sixteen consecutive bytes without assuming 16-byte alignment.
        // The short path or scan bounds guarantee all sixteen are readable.
        let mut mask = mask16(low, high, _mm_loadu_si128(ptr.cast()));
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
        let mut start = 16 - (ptr as usize & 0b00001111);
        let last = len - 16;
        while start < last {
            // Load sixteen bytes from the 16-byte-aligned interior address;
            // the loop excludes positions without a complete input vector.
            let mut mask = mask16(low, high, _mm_load_si128(ptr.add(start).cast()));
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
        let mut mask = mask16(low, high, _mm_loadu_si128(ptr.add(last).cast()));
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
pub(in super::super) fn rshufti_ssse3<const REVERT: bool>(
    low: &[u8; 16],
    high: &[u8; 16],
    haystack: &[u8],
) -> usize {
    if haystack.len() < 8 {
        return super::super::rsearch_fallback::<REVERT>(low, high, haystack);
    }
    if haystack.len() < 16 {
        // SAFETY: SSSE3 is enabled, the table arrays contain sixteen bytes,
        // and the checks above establish 8 <= haystack.len() < 16.
        return unsafe {
            rsearch_short::<REVERT>(
                _mm_loadu_si128(low.as_ptr().cast()),
                _mm_loadu_si128(high.as_ptr().cast()),
                haystack,
            )
        };
    }
    // SAFETY: array loads stay within their initialized storage; dispatch
    // validates CPU support; short inputs are handled before full input loads.
    unsafe {
        rsearch128::<REVERT>(
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
unsafe fn rsearch128<const REVERT: bool>(low: __m128i, high: __m128i, haystack: &[u8]) -> usize {
    // SAFETY: the entry point establishes SSSE3 support and len >= 16.
    // Aligned interior loads and the final overlapping vector stay in haystack.
    unsafe {
        let ptr = haystack.as_ptr();
        let len = haystack.len();
        let last = len - 16;
        // Load sixteen consecutive bytes without assuming 16-byte alignment.
        // The short path or scan bounds guarantee all sixteen are readable.
        let mut mask = mask16(low, high, _mm_loadu_si128(ptr.add(last).cast()));
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
        // Subtract that remainder from len to round ptr + end down to a
        // 16-byte boundary. The initial suffix scan covered the skipped bytes.
        let mut end = len - (ptr.add(len) as usize & 0b00001111);
        // The source compares buf_end with buf + 16 before decrementing.
        while end > 16 {
            end -= 16;
            // Load sixteen bytes from the 16-byte-aligned interior address;
            // the loop excludes positions without a complete input vector.
            let mut mask = mask16(low, high, _mm_load_si128(ptr.add(end).cast()));
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
        let mut mask = mask16(low, high, _mm_loadu_si128(ptr.cast()));
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

/// Finds the first member, or the first nonmember when `REVERT` is true.
/// Returns `haystack.len()` if no such byte exists.
///
/// # Safety
/// The CPU must support SSSE3, and `haystack` must contain 8 to 15 bytes.
#[inline(always)]
pub(super) unsafe fn search_short<const REVERT: bool>(
    low: __m128i,
    high: __m128i,
    haystack: &[u8],
) -> usize {
    // SAFETY: the caller establishes SSSE3 support and 8 <= len < 16.
    let mut mask = unsafe { mask16_overlap(low, high, haystack) };
    if !REVERT {
        // Inversion selects members and sets all bits above the input.
        // Without a real match, the first set bit is exactly the end sentinel.
        mask = !mask;
    }
    if mask != 0 {
        mask.trailing_zeros() as usize
    } else {
        haystack.len()
    }
}

/// Finds the last member, or the last nonmember when `REVERT` is true.
/// Returns `haystack.len()` if no such byte exists.
///
/// # Safety
/// The CPU must support SSSE3, and `haystack` must contain 8 to 15 bytes.
#[inline(always)]
pub(super) unsafe fn rsearch_short<const REVERT: bool>(
    low: __m128i,
    high: __m128i,
    haystack: &[u8],
) -> usize {
    // SAFETY: the caller establishes SSSE3 support and 8 <= len < 16.
    let mut mask = unsafe { mask16_overlap(low, high, haystack) };
    if !REVERT {
        // Reverse search must clear the high bits introduced by inversion;
        // otherwise it would select a position beyond the input.
        mask = !mask & ((1u16 << haystack.len()) - 1);
    }
    if mask != 0 {
        (15 - mask.leading_zeros()) as usize
    } else {
        haystack.len()
    }
}

/// Returns one nonmember bit per input byte, with higher bits clear.
///
/// # Safety
/// The CPU must support SSSE3, and `haystack` must contain 8 to 15 bytes.
#[inline(always)]
unsafe fn mask16_overlap(low: __m128i, high: __m128i, haystack: &[u8]) -> u16 {
    // SAFETY: each load reads exactly eight bytes inside haystack and permits
    // unaligned addresses. Both halves of the combined vector are initialized.
    unsafe {
        let ptr = haystack.as_ptr();
        let last = haystack.len() - 8;
        let prefix = _mm_loadl_epi64(ptr.cast());
        let suffix = _mm_loadl_epi64(ptr.add(last).cast());
        // Combine the low 64 bits of each load. For len = 15, the vector
        // contains input positions [0, ..., 7 | 7, ..., 14].
        let mask = mask16(low, high, _mm_unpacklo_epi64(prefix, suffix));
        // Keep the prefix flags; relocate suffix bits 8..15 to last..len.
        // Overlapping positions have identical flags, so OR merges them.
        (mask & 0b1111_1111) | ((mask >> 8) << last)
    }
}

#[inline(always)]
unsafe fn mask16(low: __m128i, high: __m128i, bytes: __m128i) -> u16 {
    // SAFETY: the caller has established SSSE3 support.
    unsafe {
        // set1_epi8 broadcasts 0b00001111 into every byte lane. AND with this
        // mask extracts each input byte's low nibble as a table index in 0..16.
        let nibble = _mm_set1_epi8(0b0000_1111);
        // and_si128 extracts sixteen independent low-nibble indices.
        // shuffle_epi8 (PSHUFB) selects low[index] for each byte. Its
        // index bit 7 would zero a lane, but the 0b00001111 mask keeps it clear.
        let low = _mm_shuffle_epi8(low, _mm_and_si128(bytes, nibble));
        // Look up the high nibble in the high table. andnot(a, b) means
        // (~a) & b: clear each byte's low four bits BEFORE shifting the
        // 64-bit lanes. This prevents a neighboring byte's low nibble from
        // spilling into this byte; the resulting indices are exactly b >> 4.
        let high = _mm_shuffle_epi8(high, _mm_srli_epi64::<4>(_mm_andnot_si128(nibble, bytes)));
        // Compare the intersected flags with a zero vector: 0b11111111 marks
        // a nonmember and 0b00000000 a member. movemask packs the sixteen byte
        // sign bits into the low sixteen bits of its i32 result; cast to u16.
        _mm_movemask_epi8(_mm_cmpeq_epi8(
            _mm_and_si128(low, high),
            _mm_setzero_si128(),
        )) as u16
    }
}

#[cfg(test)]
mod tests {
    use super::{_mm_loadu_si128, mask16_overlap, rsearch_short, search_short};
    use crate::{shufti::ShuftiTable, SimdBackend};

    #[test]
    fn search_short_returns_len_when_no_bytes_match() {
        if !SimdBackend::Ssse3.is_available() {
            return;
        }
        let table = ShuftiTable::new(b"a").unwrap();
        let haystack = [b'!'; 15];
        for len in 8..16 {
            // SAFETY: SSSE3 is available, the table arrays contain sixteen
            // initialized bytes, and the input has 8..15 bytes.
            let actual = unsafe {
                search_short::<false>(
                    _mm_loadu_si128(table.low().as_ptr().cast()),
                    _mm_loadu_si128(table.high().as_ptr().cast()),
                    &haystack[..len],
                )
            };
            assert_eq!(actual, len);
        }
    }

    #[test]
    fn rsearch_short_excludes_bits_past_the_input() {
        if !SimdBackend::Ssse3.is_available() {
            return;
        }
        let table = ShuftiTable::new(b"a").unwrap();
        // Every real byte matches; inversion must not introduce a match at 15.
        let haystack = [b'a'; 15];
        // SAFETY: SSSE3 is available, both table loads are in bounds, and
        // the fifteen-byte input meets the short-search length requirement.
        let actual = unsafe {
            rsearch_short::<false>(
                _mm_loadu_si128(table.low().as_ptr().cast()),
                _mm_loadu_si128(table.high().as_ptr().cast()),
                &haystack,
            )
        };
        assert_eq!(actual, 14);
    }

    #[test]
    fn mask16_overlap_preserves_each_nonmembers_position() {
        if !SimdBackend::Ssse3.is_available() {
            return;
        }
        let table = ShuftiTable::new(b"a").unwrap();
        for len in 8..16 {
            for position in 0..len {
                let mut haystack = [b'a'; 15];
                haystack[position] = b'!';
                // SAFETY: SSSE3 is available; table loads are in bounds and
                // each input slice has 8..15 initialized bytes.
                let mask = unsafe {
                    mask16_overlap(
                        _mm_loadu_si128(table.low().as_ptr().cast()),
                        _mm_loadu_si128(table.high().as_ptr().cast()),
                        &haystack[..len],
                    )
                };
                assert_eq!(mask, 1 << position, "len={len}, byte={position}");
            }
        }
    }

    #[test]
    fn mask16_overlap_clears_bits_past_the_input() {
        if !SimdBackend::Ssse3.is_available() {
            return;
        }
        let table = ShuftiTable::new(b"a").unwrap();
        // Fifteen nonmembers set bits 0..14; the duplicate lane adds no bit.
        let haystack = [b'!'; 15];
        // SAFETY: SSSE3 is available; both table loads and the two eight-byte
        // overlapping loads stay inside their respective arrays.
        let mask = unsafe {
            mask16_overlap(
                _mm_loadu_si128(table.low().as_ptr().cast()),
                _mm_loadu_si128(table.high().as_ptr().cast()),
                &haystack,
            )
        };
        assert_eq!(mask, 0b0111_1111_1111_1111);
    }
}
