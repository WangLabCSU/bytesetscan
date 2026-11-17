#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::ssse3::{rsearch_short, search_short};

/// # Safety
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(in super::super) fn truffle_avx2<const REVERT: bool>(
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
unsafe fn search<const REVERT: bool>(
    lower: __m128i,
    upper: __m128i,
    haystack: &[u8],
) -> Option<usize> {
    // SAFETY: the entry point establishes AVX2 support and len >= 4.
    // The short path is handled before full-width loads.
    // Aligned interior loads and the final overlapping vector stay in haystack.
    unsafe {
        let ptr = haystack.as_ptr();
        let len = haystack.len();
        if len < 16 {
            // SAFETY: 4 <= len < 16. AVX2 provides the
            // SSSE3 instructions used by this shared, inlined short-input scan.
            return search_short::<REVERT>(lower, upper, haystack);
        }
        if len < 32 {
            // SAFETY: the preceding branch excludes len < 16. Both overlapping
            // 16-byte loads fit in this 16..31-byte input; AVX2 is enabled.
            let mask = mask32_overlap::<REVERT>(lower, upper, haystack);
            return if mask != 0 {
                // The lowest set bit identifies the first selected lane.
                Some(mask.trailing_zeros() as usize)
            } else {
                None
            };
        }

        // Read the 32 bytes starting here without requiring alignment.
        // Overlapping end loads still end exactly at the slice boundary.
        let mask = mask32::<REVERT>(lower, upper, _mm256_loadu_si256(ptr.cast()));
        if mask != 0 {
            // The lowest set bit identifies the first selected lane.
            return Some(mask.trailing_zeros() as usize);
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
            let mask = mask32::<REVERT>(lower, upper, _mm256_load_si256(ptr.add(start).cast()));
            if mask != 0 {
                // The lowest set bit identifies the first selected lane.
                // Add this block's start to return a position in the original input.
                return Some(start + mask.trailing_zeros() as usize);
            }
            start += 32;
        }
        // Read the 32 bytes starting here without requiring alignment.
        // Overlapping end loads still end exactly at the slice boundary.
        let mask = mask32::<REVERT>(lower, upper, _mm256_loadu_si256(ptr.add(last).cast()));
        if mask != 0 {
            // The lowest set bit identifies the first selected lane.
            // Add this block's start to return a position in the original input.
            Some(last + mask.trailing_zeros() as usize)
        } else {
            None
        }
    }
}

// Reverse entry point. Requires avx2 any slice length.
#[target_feature(enable = "avx2")]
pub(in super::super) fn rtruffle_avx2<const REVERT: bool>(
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
unsafe fn rsearch<const REVERT: bool>(
    lower: __m128i,
    upper: __m128i,
    haystack: &[u8],
) -> Option<usize> {
    // SAFETY: the entry point establishes AVX2 support and len >= 4.
    // The short path is handled before full-width loads.
    // Aligned interior loads and the final overlapping vector stay in haystack.
    unsafe {
        let ptr = haystack.as_ptr();
        let len = haystack.len();
        if len < 16 {
            // SAFETY: 4 <= len < 16. AVX2 provides the
            // SSSE3 instructions used by this shared, inlined short-input scan.
            return rsearch_short::<REVERT>(lower, upper, haystack);
        }
        if len < 32 {
            // SAFETY: the preceding branch excludes len < 16. Both overlapping
            // 16-byte loads fit in this 16..31-byte input; AVX2 is enabled.
            let mask = mask32_overlap::<REVERT>(lower, upper, haystack);
            return if mask != 0 {
                // 31 - leading_zeros locates the highest set bit in this u32,
                // which identifies the last selected byte lane.
                Some((31 - mask.leading_zeros()) as usize)
            } else {
                None
            };
        }

        let last = len - 32;
        // Read the 32 bytes starting here without requiring alignment.
        // Overlapping end loads still end exactly at the slice boundary.
        let mask = mask32::<REVERT>(lower, upper, _mm256_loadu_si256(ptr.add(last).cast()));
        if mask != 0 {
            // 31 - leading_zeros locates the highest set bit in this u32,
            // which identifies the last selected byte lane.
            // Add this block's start to return a position in the original input.
            return Some(last + (31 - mask.leading_zeros()) as usize);
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
            let mask = mask32::<REVERT>(lower, upper, _mm256_load_si256(ptr.add(end).cast()));
            if mask != 0 {
                // 31 - leading_zeros locates the highest set bit in this u32,
                // which identifies the last selected byte lane.
                // Add this block's start to return a position in the original input.
                return Some(end + (31 - mask.leading_zeros()) as usize);
            }
        }
        // Read the 32 bytes starting here without requiring alignment.
        // Overlapping end loads still end exactly at the slice boundary.
        let mask = mask32::<REVERT>(lower, upper, _mm256_loadu_si256(ptr.cast()));
        if mask != 0 {
            // 31 - leading_zeros locates the highest set bit in this u32,
            // which identifies the last selected byte lane.
            Some((31 - mask.leading_zeros()) as usize)
        } else {
            None
        }
    }
}

/// Returns one selected bit per input byte, with higher bits clear.
/// `REVERT` selects nonmembers; otherwise, the mask selects members.
///
/// # Safety
/// The CPU must support AVX2, and `haystack` must contain 16 to 31 bytes.
#[inline(always)]
unsafe fn mask32_overlap<const REVERT: bool>(
    lower: __m128i,
    upper: __m128i,
    haystack: &[u8],
) -> u32 {
    // SAFETY: the caller establishes AVX2 support and 16 <= len < 32.
    // Both unaligned loads read exactly sixteen initialized bytes, covering
    // haystack[0..16] and haystack[len - 16..len] without crossing either bound.
    unsafe {
        let last = haystack.len() - 16;
        let prefix = _mm_loadu_si128(haystack.as_ptr().cast());
        let suffix = _mm_loadu_si128(haystack.as_ptr().add(last).cast());
        // The cast places prefix in the lower 128 bits. Inserting suffix into
        // lane 1 supplies the upper 128 bits, initializing all 32 byte lanes.
        // For 31 bytes, their input positions are [0, ..., 15 | 15, ..., 30].
        let bytes = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(prefix), suffix);
        let mask = mask32::<REVERT>(lower, upper, bytes);
        // Bits 0..15 already refer to the prefix's original positions. Move
        // the upper sixteen flags to the suffix's starting position, len - 16.
        // Duplicate bytes have identical flags, so OR merges their results;
        // the highest possible result bit is last + 15 == len - 1. Thus
        // bits beyond the input remain clear for either predicate.
        (mask & 0b1111_1111_1111_1111) | ((mask >> 16) << last)
    }
}

// Bit i selects byte i: members for find, nonmembers when REVERT is true.
#[inline(always)]
unsafe fn mask32<const REVERT: bool>(lower: __m128i, upper: __m128i, bytes: __m256i) -> u32 {
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
        // selected contains exactly one nonzero bit in every lane. AND above
        // therefore leaves either zero or that same bit, including bit 7.
        // Compare with zero for nonmembers, or with selected for members.
        // Both paths produce all-one/all-zero bytes without a separate NOT.
        let matches = if REVERT {
            _mm256_cmpeq_epi8(member, _mm256_setzero_si256())
        } else {
            _mm256_cmpeq_epi8(member, selected)
        };
        // Pack the top bit of each comparison result into its byte-position bit.
        _mm256_movemask_epi8(matches) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::{
        _mm256_set1_epi8, _mm_loadu_si128, _mm_set1_epi8, mask32, mask32_overlap, rsearch, search,
    };
    use crate::{truffle::TruffleTable, SimdBackend};

    #[test]
    fn mask32_selects_the_top_membership_bit_in_either_table() {
        if !SimdBackend::Avx2.is_available() {
            return;
        }
        // 0x70 and 0xf0 use different tables, but both select membership bit 7.
        for byte in [0b0111_0000u8, 0b1111_0000] {
            // SAFETY: AVX2 is available; all vector values are initialized.
            let actual = unsafe {
                let table = _mm_set1_epi8(0b1000_0000u8 as i8);
                mask32::<false>(table, table, _mm256_set1_epi8(byte as i8))
            };
            assert_eq!(actual, u32::MAX, "every lane must select byte {byte:#010b}");
        }
    }

    #[test]
    fn search_returns_none_when_the_overlapping_input_has_no_member() {
        if !SimdBackend::Avx2.is_available() {
            return;
        }
        let table = TruffleTable::new(b"a");
        let haystack = [b'!'; 31];
        for len in 16..=31 {
            // SAFETY: AVX2 is available, the table loads cover complete arrays,
            // and search handles each slice length with in-bounds loads.
            let position = unsafe {
                search::<false>(
                    _mm_loadu_si128(table.lower().as_ptr().cast()),
                    _mm_loadu_si128(table.upper().as_ptr().cast()),
                    &haystack[..len],
                )
            };
            // No 'a' exists; unused mask bits must not produce a match.
            assert_eq!(position, None);
        }
    }

    #[test]
    fn rsearch_excludes_bits_past_the_overlapping_input() {
        if !SimdBackend::Avx2.is_available() {
            return;
        }
        let table = TruffleTable::new(b"a");
        let haystack = [b'a'; 31];
        for len in 16..=31 {
            // SAFETY: AVX2 is available, both arrays contain sixteen entries,
            // and rsearch handles the input length before its vector loads.
            let position = unsafe {
                rsearch::<false>(
                    _mm_loadu_si128(table.lower().as_ptr().cast()),
                    _mm_loadu_si128(table.upper().as_ptr().cast()),
                    &haystack[..len],
                )
            };
            // Inversion must not make an out-of-input bit the last match.
            assert_eq!(position, Some(len - 1));
        }
    }

    #[test]
    fn mask32_overlap_preserves_each_nonmembers_position() {
        if !SimdBackend::Avx2.is_available() {
            return;
        }
        let table = TruffleTable::new(b"a");
        // Vary the overlap from sixteen duplicate bytes down to one. Only '!'
        // is a nonmember, so exactly its original position must be set.
        for len in 16..=31 {
            for position in 0..len {
                let mut haystack = [b'a'; 31];
                haystack[position] = b'!';
                // SAFETY: AVX2 is available, both table loads cover complete
                // arrays, and the classifier receives between 16 and 31 bytes.
                let mask = unsafe {
                    mask32_overlap::<true>(
                        _mm_loadu_si128(table.lower().as_ptr().cast()),
                        _mm_loadu_si128(table.upper().as_ptr().cast()),
                        &haystack[..len],
                    )
                };
                assert_eq!(mask, 1u32 << position, "len={len}, position={position}");
            }
        }
    }

    #[test]
    fn mask32_overlap_preserves_each_members_position() {
        if !SimdBackend::Avx2.is_available() {
            return;
        }
        let table = TruffleTable::new(b"a");
        for len in 16..32 {
            for position in 0..len {
                let mut haystack = [b'!'; 31];
                haystack[position] = b'a';
                // SAFETY: AVX2 is available, both tables are complete, and
                // each slice bounds both overlapping 16-byte loads.
                let mask = unsafe {
                    mask32_overlap::<false>(
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
    fn mask32_overlap_clears_bits_past_the_input() {
        if !SimdBackend::Avx2.is_available() {
            return;
        }
        let table = TruffleTable::new(b"a");
        let haystack = [b'!'; 31];
        // SAFETY: AVX2 is available, both table arrays contain sixteen bytes,
        // and this 31-byte input satisfies the overlapping load requirements.
        let mask = unsafe {
            mask32_overlap::<true>(
                _mm_loadu_si128(table.lower().as_ptr().cast()),
                _mm_loadu_si128(table.upper().as_ptr().cast()),
                &haystack,
            )
        };
        // All 31 input bytes are nonmembers. Suffix flags move from SIMD lanes
        // 16..31 to input positions 15..30, leaving result bit 31 clear.
        assert_eq!(mask, 0b0111_1111_1111_1111_1111_1111_1111_1111);
    }
}
