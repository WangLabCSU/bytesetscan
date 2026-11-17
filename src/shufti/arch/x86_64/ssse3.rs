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
) -> Option<usize> {
    if haystack.len() < 4 {
        return super::super::search_fallback::<REVERT>(low, high, haystack);
    }
    if haystack.len() < 16 {
        // SAFETY: SSSE3 is enabled, the table arrays contain sixteen bytes,
        // and the checks above establish 4 <= haystack.len() < 16.
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
unsafe fn search128<const REVERT: bool>(
    low: __m128i,
    high: __m128i,
    haystack: &[u8],
) -> Option<usize> {
    // SAFETY: the entry point establishes SSSE3 support and len >= 16.
    // Aligned interior loads and the final overlapping vector stay in haystack.
    unsafe {
        let ptr = haystack.as_ptr();
        let len = haystack.len();
        // Load sixteen consecutive bytes without assuming 16-byte alignment.
        // The short path or scan bounds guarantee all sixteen are readable.
        let mask = mask16::<REVERT>(low, high, _mm_loadu_si128(ptr.cast()));
        if mask != 0 {
            // The lowest set bit identifies the first selected lane.
            return Some(mask.trailing_zeros() as usize);
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
            let mask = mask16::<REVERT>(low, high, _mm_load_si128(ptr.add(start).cast()));
            if mask != 0 {
                // The lowest set bit identifies the first selected lane.
                // Add this block's start to return a position in the original input.
                return Some(start + mask.trailing_zeros() as usize);
            }
            start += 16;
        }
        // Load sixteen consecutive bytes without assuming 16-byte alignment.
        // The short path or scan bounds guarantee all sixteen are readable.
        let mask = mask16::<REVERT>(low, high, _mm_loadu_si128(ptr.add(last).cast()));
        if mask != 0 {
            // The lowest set bit identifies the first selected lane.
            // Add this block's start to return a position in the original input.
            Some(last + mask.trailing_zeros() as usize)
        } else {
            None
        }
    }
}

// Reverse entry point. Requires ssse3 any slice length.
#[target_feature(enable = "ssse3")]
pub(in super::super) fn rshufti_ssse3<const REVERT: bool>(
    low: &[u8; 16],
    high: &[u8; 16],
    haystack: &[u8],
) -> Option<usize> {
    if haystack.len() < 4 {
        return super::super::rsearch_fallback::<REVERT>(low, high, haystack);
    }
    if haystack.len() < 16 {
        // SAFETY: SSSE3 is enabled, the table arrays contain sixteen bytes,
        // and the checks above establish 4 <= haystack.len() < 16.
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
unsafe fn rsearch128<const REVERT: bool>(
    low: __m128i,
    high: __m128i,
    haystack: &[u8],
) -> Option<usize> {
    // SAFETY: the entry point establishes SSSE3 support and len >= 16.
    // Aligned interior loads and the final overlapping vector stay in haystack.
    unsafe {
        let ptr = haystack.as_ptr();
        let len = haystack.len();
        let last = len - 16;
        // Load sixteen consecutive bytes without assuming 16-byte alignment.
        // The short path or scan bounds guarantee all sixteen are readable.
        let mask = mask16::<REVERT>(low, high, _mm_loadu_si128(ptr.add(last).cast()));
        if mask != 0 {
            return Some(last + (31 - (mask as u32).leading_zeros()) as usize);
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
            let mask = mask16::<REVERT>(low, high, _mm_load_si128(ptr.add(end).cast()));
            if mask != 0 {
                return Some(end + (31 - (mask as u32).leading_zeros()) as usize);
            }
        }
        // Load sixteen consecutive bytes without assuming 16-byte alignment.
        // The short path or scan bounds guarantee all sixteen are readable.
        let mask = mask16::<REVERT>(low, high, _mm_loadu_si128(ptr.cast()));
        if mask != 0 {
            Some((31 - (mask as u32).leading_zeros()) as usize)
        } else {
            None
        }
    }
}

/// Finds the first member, or the first nonmember when `REVERT` is true.
/// Returns `None` if no such byte exists.
///
/// # Safety
/// The CPU must support SSSE3, and `haystack` must contain 4 to 15 bytes.
#[inline(always)]
pub(super) unsafe fn search_short<const REVERT: bool>(
    low: __m128i,
    high: __m128i,
    haystack: &[u8],
) -> Option<usize> {
    if haystack.len() < 8 {
        // SAFETY: the caller establishes SSSE3 support and len >= 4;
        // this branch bounds both overlapping four-byte loads.
        let mask = unsafe { mask8_overlap::<REVERT>(low, high, haystack) };
        return if mask != 0 {
            Some(mask.trailing_zeros() as usize)
        } else {
            None
        };
    }
    // SAFETY: SSSE3 is available and the preceding branch establishes 8 <= len < 16.
    let mask = unsafe { mask16_overlap::<REVERT>(low, high, haystack) };
    if mask != 0 {
        Some(mask.trailing_zeros() as usize)
    } else {
        None
    }
}

/// Finds the last member, or the last nonmember when `REVERT` is true.
/// Returns `None` if no such byte exists.
///
/// # Safety
/// The CPU must support SSSE3, and `haystack` must contain 4 to 15 bytes.
#[inline(always)]
pub(super) unsafe fn rsearch_short<const REVERT: bool>(
    low: __m128i,
    high: __m128i,
    haystack: &[u8],
) -> Option<usize> {
    if haystack.len() < 8 {
        // SAFETY: the caller establishes SSSE3 support and len >= 4;
        // this branch bounds both overlapping four-byte loads.
        let mask = unsafe { mask8_overlap::<REVERT>(low, high, haystack) };
        return if mask != 0 {
            Some((7 - mask.leading_zeros()) as usize)
        } else {
            None
        };
    }
    // SAFETY: SSSE3 is available and the preceding branch establishes 8 <= len < 16.
    let mask = unsafe { mask16_overlap::<REVERT>(low, high, haystack) };
    if mask != 0 {
        Some((15 - mask.leading_zeros()) as usize)
    } else {
        None
    }
}

/// Returns one selected bit per input byte, with higher bits clear.
/// `REVERT` selects nonmembers; otherwise, the mask selects members.
///
/// # Safety
/// The CPU must support SSSE3, and `haystack` must contain 4 to 7 bytes.
#[inline(always)]
unsafe fn mask8_overlap<const REVERT: bool>(low: __m128i, high: __m128i, haystack: &[u8]) -> u8 {
    // SAFETY: both unaligned loads read exactly four initialized bytes inside
    // haystack. Each load zeroes the rest of its register; unpack initializes
    // all sixteen lanes before mask16 classifies them.
    unsafe {
        let ptr = haystack.as_ptr();
        let last = haystack.len() - 4;
        let prefix = _mm_loadu_si32(ptr.cast());
        let suffix = _mm_loadu_si32(ptr.add(last).cast());
        // Interleave the low 32-bit lanes. For len = 7, the first eight bytes
        // are input positions [0, 1, 2, 3 | 3, 4, 5, 6]; the rest are zero.
        let mask = mask16::<REVERT>(low, high, _mm_unpacklo_epi32(prefix, suffix)) as u8;
        // Casting discards flags from the eight zero-filled lanes. Keep
        // prefix bits 0..3, then relocate suffix bits 4..7 to last..len.
        // Duplicate input positions have identical flags, so OR merges them.
        // The suffix's highest bit lands at last + 3 == len - 1, leaving
        // bits beyond the input clear for either predicate.
        (mask & 0b0000_1111) | ((mask >> 4) << last)
    }
}

/// Returns one selected bit per input byte, with higher bits clear.
/// `REVERT` selects nonmembers; otherwise, the mask selects members.
///
/// # Safety
/// The CPU must support SSSE3, and `haystack` must contain 8 to 15 bytes.
#[inline(always)]
unsafe fn mask16_overlap<const REVERT: bool>(low: __m128i, high: __m128i, haystack: &[u8]) -> u16 {
    // SAFETY: each load reads exactly eight bytes inside haystack and permits
    // unaligned addresses. Both halves of the combined vector are initialized.
    unsafe {
        let ptr = haystack.as_ptr();
        let last = haystack.len() - 8;
        let prefix = _mm_loadl_epi64(ptr.cast());
        let suffix = _mm_loadl_epi64(ptr.add(last).cast());
        // Combine the low 64 bits of each load. For len = 15, the vector
        // contains input positions [0, ..., 7 | 7, ..., 14].
        let mask = mask16::<REVERT>(low, high, _mm_unpacklo_epi64(prefix, suffix));
        // Keep the prefix flags; relocate suffix bits 8..15 to last..len.
        // Overlapping positions have identical flags, so OR merges them.
        // The suffix's highest bit lands at last + 7 == len - 1, leaving
        // bits beyond the input clear for either predicate.
        (mask & 0b1111_1111) | ((mask >> 8) << last)
    }
}

// Bit i selects byte i: members for find, nonmembers when REVERT is true.
// Each high-table entry must be zero or contain exactly one group flag.
#[inline(always)]
unsafe fn mask16<const REVERT: bool>(low: __m128i, high: __m128i, bytes: __m128i) -> u16 {
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
        // A nonzero intersection means membership. Only each byte's top bit
        // is consumed by movemask, so member flags need not become all ones.
        let member = _mm_and_si128(low, high);
        let selected = if REVERT {
            // Zero becomes 0b11111111; nonzero becomes 0b00000000.
            _mm_cmpeq_epi8(member, _mm_setzero_si128())
        } else {
            // ShuftiTable gives each high nibble at most one group flag.
            // The intersection is therefore zero or a power of two <= 128.
            // Adding 127 cannot overflow: zero becomes 127 (top bit clear),
            // and a member becomes 128..255 (top bit set). Other output bits
            // are irrelevant to movemask; no packed-mask inversion is needed.
            _mm_add_epi8(member, _mm_set1_epi8(0b0111_1111))
        };
        // Pack each byte's top bit into bit i, then retain exactly sixteen bits.
        _mm_movemask_epi8(selected) as u16
    }
}

#[cfg(test)]
mod tests {
    use super::{
        _mm_loadu_si128, _mm_set1_epi8, _mm_setzero_si128, mask16, mask16_overlap, mask8_overlap,
        rsearch_short, search_short,
    };
    use crate::{shufti::ShuftiTable, SimdBackend};

    #[test]
    fn mask16_selects_each_group_flag() {
        if !SimdBackend::Ssse3.is_available() {
            return;
        }
        // A high-table entry selects one of eight groups. Include bit 7,
        // which a signed greater-than-zero comparison would incorrectly reject.
        for group in 0..8 {
            let flag = 1u8 << group;
            // SAFETY: SSSE3 is available; the initialized vectors need no loads.
            let actual = unsafe {
                let table = _mm_set1_epi8(flag as i8);
                mask16::<false>(table, table, _mm_setzero_si128())
            };
            assert_eq!(actual, u16::MAX, "every lane must select flag {flag:#010b}");
        }
    }

    #[test]
    fn search_short_returns_none_when_no_bytes_match() {
        if !SimdBackend::Ssse3.is_available() {
            return;
        }
        let table = ShuftiTable::new(b"a").unwrap();
        let haystack = [b'!'; 15];
        for len in 4..16 {
            // SAFETY: SSSE3 is available, the table arrays contain sixteen
            // initialized bytes, and the input has 4..15 bytes.
            let actual = unsafe {
                search_short::<false>(
                    _mm_loadu_si128(table.low().as_ptr().cast()),
                    _mm_loadu_si128(table.high().as_ptr().cast()),
                    &haystack[..len],
                )
            };
            assert_eq!(actual, None);
        }
    }

    #[test]
    fn rsearch_short_excludes_bits_past_the_input() {
        if !SimdBackend::Ssse3.is_available() {
            return;
        }
        let table = ShuftiTable::new(b"a").unwrap();
        // Every real byte matches; inversion must not introduce a match at len.
        let haystack = [b'a'; 15];
        for len in 4..16 {
            // SAFETY: SSSE3 is available, both table loads are in bounds, and
            // each input slice meets the 4..15-byte short-search requirement.
            let actual = unsafe {
                rsearch_short::<false>(
                    _mm_loadu_si128(table.low().as_ptr().cast()),
                    _mm_loadu_si128(table.high().as_ptr().cast()),
                    &haystack[..len],
                )
            };
            assert_eq!(actual, Some(len - 1));
        }
    }

    #[test]
    fn mask8_overlap_preserves_each_nonmembers_position() {
        if !SimdBackend::Ssse3.is_available() {
            return;
        }
        let table = ShuftiTable::new(b"a").unwrap();
        for len in 4..8 {
            for position in 0..len {
                let mut haystack = [b'a'; 7];
                haystack[position] = b'!';
                // SAFETY: SSSE3 is available; table loads are in bounds and
                // each input slice has 4..7 initialized bytes.
                let mask = unsafe {
                    mask8_overlap::<true>(
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
    fn mask8_overlap_preserves_each_members_position() {
        if !SimdBackend::Ssse3.is_available() {
            return;
        }
        let table = ShuftiTable::new(b"a").unwrap();
        for len in 4..8 {
            for position in 0..len {
                let mut haystack = [b'!'; 7];
                haystack[position] = b'a';
                // SAFETY: SSSE3 is available, table loads are in bounds, and
                // each 4..7-byte slice bounds both overlapping four-byte loads.
                let mask = unsafe {
                    mask8_overlap::<false>(
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
    fn mask8_overlap_excludes_matching_zero_filled_lanes() {
        if !SimdBackend::Ssse3.is_available() {
            return;
        }
        let table = ShuftiTable::new(&[0]).unwrap();
        // Only zero matches. The input has no zero, but the SIMD load fills
        // unused lanes with it; selecting members must still yield no bits.
        let haystack = [b'!'; 7];
        for len in 4..8 {
            // SAFETY: SSSE3 is available, table loads are in bounds, and
            // each slice has the four bytes required by both loads.
            let mask = unsafe {
                mask8_overlap::<false>(
                    _mm_loadu_si128(table.low().as_ptr().cast()),
                    _mm_loadu_si128(table.high().as_ptr().cast()),
                    &haystack[..len],
                )
            };
            assert_eq!(mask, 0, "len={len}");
        }
    }

    #[test]
    fn mask8_overlap_excludes_zero_filled_lanes() {
        if !SimdBackend::Ssse3.is_available() {
            return;
        }
        let table = ShuftiTable::new(b"a").unwrap();
        // All input bytes are members; zero-filled lanes are nonmembers.
        // Their flags must not leak into the returned input mask.
        let haystack = [b'a'; 7];
        // SAFETY: SSSE3 is available, table loads are in bounds, and the
        // seven-byte input bounds both overlapping four-byte loads.
        let mask = unsafe {
            mask8_overlap::<true>(
                _mm_loadu_si128(table.low().as_ptr().cast()),
                _mm_loadu_si128(table.high().as_ptr().cast()),
                &haystack,
            )
        };
        assert_eq!(mask, 0);
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
                    mask16_overlap::<true>(
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
    fn mask16_overlap_preserves_each_members_position() {
        if !SimdBackend::Ssse3.is_available() {
            return;
        }
        let table = ShuftiTable::new(b"a").unwrap();
        for len in 8..16 {
            for position in 0..len {
                let mut haystack = [b'!'; 15];
                haystack[position] = b'a';
                // SAFETY: SSSE3 is available, both tables are complete, and
                // each slice bounds both overlapping 8-byte loads.
                let mask = unsafe {
                    mask16_overlap::<false>(
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
            mask16_overlap::<true>(
                _mm_loadu_si128(table.low().as_ptr().cast()),
                _mm_loadu_si128(table.high().as_ptr().cast()),
                &haystack,
            )
        };
        assert_eq!(mask, 0b0111_1111_1111_1111);
    }
}
