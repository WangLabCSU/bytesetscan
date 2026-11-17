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
) -> Option<usize> {
    // Two overlapping four-byte loads require at least four input bytes.
    if haystack.len() < 4 {
        return super::super::search_fallback::<REVERT>(lower, upper, haystack);
    }
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
unsafe fn search128<const REVERT: bool>(
    lower: __m128i,
    upper: __m128i,
    haystack: &[u8],
) -> Option<usize> {
    // SAFETY: the entry point establishes SSSE3 support and len >= 4.
    // The short path is handled before full-width loads.
    // Aligned interior loads and the final overlapping vector stay in haystack.
    unsafe {
        let ptr = haystack.as_ptr();
        let len = haystack.len();
        if len < 16 {
            // SAFETY: SSSE3 is enabled and 4 <= len < 16.
            return search_short::<REVERT>(lower, upper, haystack);
        }
        // Load sixteen consecutive bytes without assuming 16-byte alignment.
        // The short path or scan bounds guarantee all sixteen are readable.
        let mut mask = mask16(lower, upper, _mm_loadu_si128(ptr.cast()));
        if !REVERT {
            // Convert the classifier's nonmember bits into member bits. This
            // changes the predicate only; forward/reverse bit ordering is unchanged.
            mask = !mask;
        }
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
            let mut mask = mask16(lower, upper, _mm_load_si128(ptr.add(start).cast()));
            if !REVERT {
                // Convert the classifier's nonmember bits into member bits. This
                // changes the predicate only; forward/reverse bit ordering is unchanged.
                mask = !mask;
            }
            if mask != 0 {
                // The lowest set bit identifies the first selected lane.
                // Add this block's start to return a position in the original input.
                return Some(start + mask.trailing_zeros() as usize);
            }
            start += 16;
        }
        // Load sixteen consecutive bytes without assuming 16-byte alignment.
        // The short path or scan bounds guarantee all sixteen are readable.
        let mut mask = mask16(lower, upper, _mm_loadu_si128(ptr.add(last).cast()));
        if !REVERT {
            // Convert the classifier's nonmember bits into member bits. This
            // changes the predicate only; forward/reverse bit ordering is unchanged.
            mask = !mask;
        }
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
pub(in super::super) fn rtruffle_ssse3<const REVERT: bool>(
    lower: &[u8; 16],
    upper: &[u8; 16],
    haystack: &[u8],
) -> Option<usize> {
    // Two overlapping four-byte loads require at least four input bytes.
    if haystack.len() < 4 {
        return super::super::rsearch_fallback::<REVERT>(lower, upper, haystack);
    }
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
unsafe fn rsearch128<const REVERT: bool>(
    lower: __m128i,
    upper: __m128i,
    haystack: &[u8],
) -> Option<usize> {
    // SAFETY: the entry point establishes SSSE3 support and len >= 4.
    // The short path is handled before full-width loads.
    // Aligned interior loads and the final overlapping vector stay in haystack.
    unsafe {
        let ptr = haystack.as_ptr();
        let len = haystack.len();
        if len < 16 {
            // SAFETY: SSSE3 is enabled and 4 <= len < 16.
            return rsearch_short::<REVERT>(lower, upper, haystack);
        }
        let last = len - 16;
        // Load sixteen consecutive bytes without assuming 16-byte alignment.
        // The short path or scan bounds guarantee all sixteen are readable.
        let mut mask = mask16(lower, upper, _mm_loadu_si128(ptr.add(last).cast()));
        if !REVERT {
            // Convert the classifier's nonmember bits into member bits. This
            // changes the predicate only; forward/reverse bit ordering is unchanged.
            mask = !mask;
        }
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
            let mut mask = mask16(lower, upper, _mm_load_si128(ptr.add(end).cast()));
            if !REVERT {
                // Convert the classifier's nonmember bits into member bits. This
                // changes the predicate only; forward/reverse bit ordering is unchanged.
                mask = !mask;
            }
            if mask != 0 {
                return Some(end + (31 - (mask as u32).leading_zeros()) as usize);
            }
        }
        // Load sixteen consecutive bytes without assuming 16-byte alignment.
        // The short path or scan bounds guarantee all sixteen are readable.
        let mut mask = mask16(lower, upper, _mm_loadu_si128(ptr.cast()));
        if !REVERT {
            // Convert the classifier's nonmember bits into member bits. This
            // changes the predicate only; forward/reverse bit ordering is unchanged.
            mask = !mask;
        }
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
    lower: __m128i,
    upper: __m128i,
    haystack: &[u8],
) -> Option<usize> {
    // SAFETY: SSSE3 is available and 4 <= len < 16. The length check chooses
    // four-byte loads for 4..7 bytes and eight-byte loads for 8..15 bytes.
    unsafe {
        let len = haystack.len();
        if len < 8 {
            let mask = mask8_overlap::<REVERT>(lower, upper, haystack);
            return if mask != 0 {
                Some(mask.trailing_zeros() as usize)
            } else {
                None
            };
        }
        let mask = mask16_overlap::<REVERT>(lower, upper, haystack);
        if mask != 0 {
            Some(mask.trailing_zeros() as usize)
        } else {
            None
        }
    }
}

/// Finds the last member, or the last nonmember when `REVERT` is true.
/// Returns `None` if no such byte exists.
///
/// # Safety
/// The CPU must support SSSE3, and `haystack` must contain 4 to 15 bytes.
#[inline(always)]
pub(super) unsafe fn rsearch_short<const REVERT: bool>(
    lower: __m128i,
    upper: __m128i,
    haystack: &[u8],
) -> Option<usize> {
    // SAFETY: SSSE3 is available and 4 <= len < 16. The length check chooses
    // four-byte loads for 4..7 bytes and eight-byte loads for 8..15 bytes.
    unsafe {
        let len = haystack.len();
        if len < 8 {
            let mask = mask8_overlap::<REVERT>(lower, upper, haystack);
            return if mask != 0 {
                Some((7 - mask.leading_zeros()) as usize)
            } else {
                None
            };
        }
        let mask = mask16_overlap::<REVERT>(lower, upper, haystack);
        if mask != 0 {
            Some((15 - mask.leading_zeros()) as usize)
        } else {
            None
        }
    }
}

/// Returns one selected bit per input byte, with higher bits clear.
/// `REVERT` selects nonmembers; otherwise, the mask selects members.
///
/// # Safety
/// The CPU must support SSSE3, and `haystack` must contain 4 to 7 bytes.
#[inline(always)]
unsafe fn mask8_overlap<const REVERT: bool>(lower: __m128i, upper: __m128i, haystack: &[u8]) -> u8 {
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
        let mut mask = mask16(lower, upper, _mm_unpacklo_epi32(prefix, suffix)) as u8;
        if !REVERT {
            // Select members before merging the two four-byte masks. Every
            // retained bit still describes a real byte in one of the loads.
            mask = !mask;
        }
        // Casting above discards the eight zero-filled lanes' flags. Keep
        // prefix bits 0..3, then relocate suffix bits 4..7 to last..len.
        // The repeated input positions have identical flags, so OR merges them.
        // The suffix's highest bit lands at last + 3 == len - 1. Thus even
        // after inversion, no bit can escape the actual input range.
        (mask & 0b0000_1111) | ((mask >> 4) << last)
    }
}

/// Returns one selected bit per input byte, with higher bits clear.
/// `REVERT` selects nonmembers; otherwise, the mask selects members.
///
/// # Safety
/// The CPU must support SSSE3, and `haystack` must contain 8 to 15 bytes.
#[inline(always)]
unsafe fn mask16_overlap<const REVERT: bool>(
    lower: __m128i,
    upper: __m128i,
    haystack: &[u8],
) -> u16 {
    // SAFETY: the caller establishes SSSE3 support and 8 <= len < 16.
    // The loads read exactly haystack[0..8] and haystack[len - 8..len].
    // Each reads eight bytes without requiring alignment, despite taking a
    // __m128i pointer. Their values and both table vectors are initialized,
    // so the combined vector meets mask16's requirements.
    unsafe {
        let last = haystack.len() - 8;
        let prefix = _mm_loadl_epi64(haystack.as_ptr().cast());
        let suffix = _mm_loadl_epi64(haystack.as_ptr().add(last).cast());
        // Put the prefix in the lower eight lanes and the suffix in the upper
        // eight. For 15 bytes, input positions are [0, ..., 7 | 7, ..., 14]:
        // position 7 occurs twice in the vector.
        let bytes = _mm_unpacklo_epi64(prefix, suffix);
        let mut mask = mask16(lower, upper, bytes);
        if !REVERT {
            // Select members while all sixteen bits still describe real
            // input bytes. Relocating the suffix then removes the overlap.
            mask = !mask;
        }
        // Low eight flags already have their input positions. Move the suffix
        // flags down to its actual starting position, len - 8, then combine.
        // Duplicate bytes have identical flags, so OR preserves membership.
        // The suffix's highest bit lands at last + 7 == len - 1, leaving
        // bits beyond the input clear for either predicate.
        (mask & 0b1111_1111) | ((mask >> 8) << last)
    }
}

/// Returns one nonmember bit for each byte lane, starting at bit zero.
///
/// # Safety
/// The CPU must support SSSE3.
#[inline(always)]
unsafe fn mask16(lower: __m128i, upper: __m128i, bytes: __m128i) -> u16 {
    // SAFETY: the caller establishes SSSE3 support, including when inlined
    // into an AVX2 entry point. All inputs are initialized vector values;
    // the shuffles, shifts and comparisons perform no pointer-based reads.
    unsafe {
        // set1_epi8(0b10000000u8 as i8) repeats the bit pattern 0b10000000 in every byte.
        // This is the shuffle index's zeroing bit, not a signed comparison.
        let top = _mm_set1_epi8(0b1000_0000u8 as i8);
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

#[cfg(test)]
mod tests {
    use super::{_mm_loadu_si128, mask16_overlap, mask8_overlap, rsearch_short, search_short};
    use crate::{truffle::TruffleTable, SimdBackend};

    #[test]
    fn search_short_returns_none_when_no_short_input_byte_matches() {
        if !SimdBackend::Ssse3.is_available() {
            return;
        }
        let table = TruffleTable::new(b"a");
        let haystack = [b'!'; 15];
        for len in 4..=15 {
            // SAFETY: SSSE3 is available, both table arrays contain 16 bytes,
            // and search_short handles this slice length without reading outside it.
            let position = unsafe {
                search_short::<false>(
                    _mm_loadu_si128(table.lower().as_ptr().cast()),
                    _mm_loadu_si128(table.upper().as_ptr().cast()),
                    &haystack[..len],
                )
            };
            // Inversion leaves the first set bit at len, not at a real match.
            assert_eq!(position, None);
        }
    }

    #[test]
    fn rsearch_short_returns_the_last_of_fifteen_members() {
        if !SimdBackend::Ssse3.is_available() {
            return;
        }
        let table = TruffleTable::new(b"a");
        let haystack = [b'a'; 15];
        // SAFETY: SSSE3 is available and both table arrays contain 16 bytes.
        // rsearch_short handles this short slice with in-bounds eight-byte loads.
        let position = unsafe {
            rsearch_short::<false>(
                _mm_loadu_si128(table.lower().as_ptr().cast()),
                _mm_loadu_si128(table.upper().as_ptr().cast()),
                &haystack,
            )
        };
        // Bit 15 becomes set on inversion but is outside the input.
        assert_eq!(position, Some(14));
    }

    #[test]
    fn mask8_overlap_preserves_each_nonmembers_position() {
        if !SimdBackend::Ssse3.is_available() {
            return;
        }
        let table = TruffleTable::new(b"a");
        for len in 4..8 {
            for position in 0..len {
                let mut haystack = [b'a'; 7];
                haystack[position] = b'!';
                // SAFETY: SSSE3 is available, both table loads are in bounds,
                // and the input contains 4..7 initialized bytes.
                let mask = unsafe {
                    mask8_overlap::<true>(
                        _mm_loadu_si128(table.lower().as_ptr().cast()),
                        _mm_loadu_si128(table.upper().as_ptr().cast()),
                        &haystack[..len],
                    )
                };
                assert_eq!(mask, 1 << position, "len={len}, position={position}");
            }
        }
    }

    #[test]
    fn mask8_overlap_preserves_each_members_position() {
        if !SimdBackend::Ssse3.is_available() {
            return;
        }
        let table = TruffleTable::new(b"a");
        for len in 4..8 {
            for position in 0..len {
                let mut haystack = [b'!'; 7];
                haystack[position] = b'a';
                // SAFETY: SSSE3 is available, table loads are in bounds, and
                // each 4..7-byte slice bounds both overlapping four-byte loads.
                let mask = unsafe {
                    mask8_overlap::<false>(
                        _mm_loadu_si128(table.lower().as_ptr().cast()),
                        _mm_loadu_si128(table.upper().as_ptr().cast()),
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
        let table = TruffleTable::new(&[0]);
        // Only zero matches. The input has no zero, but the SIMD load fills
        // unused lanes with it; selecting members must still yield no bits.
        let haystack = [b'!'; 7];
        for len in 4..8 {
            // SAFETY: SSSE3 is available, table loads are in bounds, and
            // each slice has the four bytes required by both loads.
            let mask = unsafe {
                mask8_overlap::<false>(
                    _mm_loadu_si128(table.lower().as_ptr().cast()),
                    _mm_loadu_si128(table.upper().as_ptr().cast()),
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
        // All input bytes are members. The eight zero-filled vector lanes
        // are nonmembers, but none of their flags may reach the result.
        let table = TruffleTable::new(b"a");
        let haystack = [b'a'; 7];
        // SAFETY: SSSE3 is available; both tables and both four-byte input
        // loads are entirely inside their initialized arrays.
        let mask = unsafe {
            mask8_overlap::<true>(
                _mm_loadu_si128(table.lower().as_ptr().cast()),
                _mm_loadu_si128(table.upper().as_ptr().cast()),
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
        let table = TruffleTable::new(b"a");
        // Vary the overlap from eight duplicate bytes down to one. Only '!'
        // is a nonmember, so exactly its original position must be set.
        for len in 8..=15 {
            for position in 0..len {
                let mut haystack = [b'a'; 15];
                haystack[position] = b'!';
                // SAFETY: SSSE3 is available, the tables each contain 16 bytes,
                // and the short classifier receives between 8 and 15 bytes.
                let mask = unsafe {
                    mask16_overlap::<true>(
                        _mm_loadu_si128(table.lower().as_ptr().cast()),
                        _mm_loadu_si128(table.upper().as_ptr().cast()),
                        &haystack[..len],
                    )
                };
                assert_eq!(mask, 1 << position, "len={len}, position={position}");
            }
        }
    }

    #[test]
    fn mask16_overlap_preserves_each_members_position() {
        if !SimdBackend::Ssse3.is_available() {
            return;
        }
        let table = TruffleTable::new(b"a");
        for len in 8..16 {
            for position in 0..len {
                let mut haystack = [b'!'; 15];
                haystack[position] = b'a';
                // SAFETY: SSSE3 is available, both tables are complete, and
                // each slice bounds both overlapping 8-byte loads.
                let mask = unsafe {
                    mask16_overlap::<false>(
                        _mm_loadu_si128(table.lower().as_ptr().cast()),
                        _mm_loadu_si128(table.upper().as_ptr().cast()),
                        &haystack[..len],
                    )
                };
                assert_eq!(mask, 1 << position, "len={len}, byte={position}");
            }
        }
    }

    #[test]
    fn mask16_overlap_excludes_the_sixteenth_lane_for_fifteen_nonmembers() {
        if !SimdBackend::Ssse3.is_available() {
            return;
        }
        let table = TruffleTable::new(b"a");
        let haystack = [b'!'; 15];
        // SAFETY: SSSE3 is available, both table loads cover initialized arrays,
        // and the 15-byte slice meets the short classifier's length contract.
        let mask = unsafe {
            mask16_overlap::<true>(
                _mm_loadu_si128(table.lower().as_ptr().cast()),
                _mm_loadu_si128(table.upper().as_ptr().cast()),
                &haystack,
            )
        };
        // Fifteen real bytes are nonmembers. The duplicated byte in the
        // sixteenth SIMD lane must not create an out-of-slice result bit.
        assert_eq!(mask, 0b0111_1111_1111_1111);
    }
}
