#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::ssse3::{rsearch_short, search_short};

/// # Safety
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(in super::super) fn shufti_avx2<const REVERT: bool>(
    low: &[u8; 16],
    high: &[u8; 16],
    haystack: &[u8],
) -> Option<usize> {
    if haystack.len() < 4 {
        return super::super::search_fallback::<REVERT>(low, high, haystack);
    }
    if haystack.len() < 16 {
        // SAFETY: AVX2 enables SSSE3; both table loads are in bounds and
        // the checks above establish the helper's 4..15-byte input range.
        return unsafe {
            search_short::<REVERT>(
                _mm_loadu_si128(low.as_ptr().cast()),
                _mm_loadu_si128(high.as_ptr().cast()),
                haystack,
            )
        };
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
unsafe fn search<const REVERT: bool>(
    low: __m128i,
    high: __m128i,
    haystack: &[u8],
) -> Option<usize> {
    // SAFETY: the entry point establishes AVX2 support and len >= 16.
    // The short path uses two in-bounds sixteen-byte loads.
    // Aligned interior loads and the final overlapping vector stay in haystack.
    unsafe {
        let ptr = haystack.as_ptr();
        let len = haystack.len();
        // shuftiFwdShort / shuftiRevShort use two overlapping 16-byte reads.
        if len <= 32 {
            // Load sixteen consecutive bytes without assuming 16-byte alignment.
            // The short path or scan bounds guarantee all sixteen are readable.
            let mask = mask16_avx2::<REVERT>(low, high, _mm_loadu_si128(ptr.cast()));
            if mask != 0 {
                // The lowest set bit identifies the first selected lane.
                return Some(mask.trailing_zeros() as usize);
            }
            let last = len - 16;
            // Load sixteen consecutive bytes without assuming 16-byte alignment.
            // The short path or scan bounds guarantee all sixteen are readable.
            let mask = mask16_avx2::<REVERT>(low, high, _mm_loadu_si128(ptr.add(last).cast()));
            return if mask != 0 {
                // The lowest set bit identifies the first selected lane.
                // Add this block's start to return a position in the original input.
                Some(last + mask.trailing_zeros() as usize)
            } else {
                None
            };
        }
        // Read the 32 bytes starting here without requiring alignment.
        // Overlapping end loads still end exactly at the slice boundary.
        let mask = mask32::<REVERT>(low, high, _mm256_loadu_si256(ptr.cast()));
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
            let mask = mask32::<REVERT>(low, high, _mm256_load_si256(ptr.add(start).cast()));
            if mask != 0 {
                // The lowest set bit identifies the first selected lane.
                // Add this block's start to return a position in the original input.
                return Some(start + mask.trailing_zeros() as usize);
            }
            start += 32;
        }
        // Read the 32 bytes starting here without requiring alignment.
        // Overlapping end loads still end exactly at the slice boundary.
        let mask = mask32::<REVERT>(low, high, _mm256_loadu_si256(ptr.add(last).cast()));
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
pub(in super::super) fn rshufti_avx2<const REVERT: bool>(
    low: &[u8; 16],
    high: &[u8; 16],
    haystack: &[u8],
) -> Option<usize> {
    if haystack.len() < 4 {
        return super::super::rsearch_fallback::<REVERT>(low, high, haystack);
    }
    if haystack.len() < 16 {
        // SAFETY: AVX2 enables SSSE3; both table loads are in bounds and
        // the checks above establish the helper's 4..15-byte input range.
        return unsafe {
            rsearch_short::<REVERT>(
                _mm_loadu_si128(low.as_ptr().cast()),
                _mm_loadu_si128(high.as_ptr().cast()),
                haystack,
            )
        };
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
unsafe fn rsearch<const REVERT: bool>(
    low: __m128i,
    high: __m128i,
    haystack: &[u8],
) -> Option<usize> {
    // SAFETY: the entry point establishes AVX2 support and len >= 16.
    // The short path uses two in-bounds sixteen-byte loads.
    // Aligned interior loads and the final overlapping vector stay in haystack.
    unsafe {
        let ptr = haystack.as_ptr();
        let len = haystack.len();
        // shuftiFwdShort / shuftiRevShort use two overlapping 16-byte reads.
        if len <= 32 {
            let last = len - 16;
            // Load sixteen consecutive bytes without assuming 16-byte alignment.
            // The short path or scan bounds guarantee all sixteen are readable.
            let mask = mask16_avx2::<REVERT>(low, high, _mm_loadu_si128(ptr.add(last).cast()));
            if mask != 0 {
                return Some(last + (31 - (mask as u32).leading_zeros()) as usize);
            }
            // Load sixteen consecutive bytes without assuming 16-byte alignment.
            // The short path or scan bounds guarantee all sixteen are readable.
            let mask = mask16_avx2::<REVERT>(low, high, _mm_loadu_si128(ptr.cast()));
            return if mask != 0 {
                Some((31 - (mask as u32).leading_zeros()) as usize)
            } else {
                None
            };
        }
        let last = len - 32;
        // Read the 32 bytes starting here without requiring alignment.
        // Overlapping end loads still end exactly at the slice boundary.
        let mask = mask32::<REVERT>(low, high, _mm256_loadu_si256(ptr.add(last).cast()));
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
            let mask = mask32::<REVERT>(low, high, _mm256_load_si256(ptr.add(end).cast()));
            if mask != 0 {
                // 31 - leading_zeros locates the highest set bit in this u32,
                // which identifies the last selected byte lane.
                // Add this block's start to return a position in the original input.
                return Some(end + (31 - mask.leading_zeros()) as usize);
            }
        }
        // Read the 32 bytes starting here without requiring alignment.
        // Overlapping end loads still end exactly at the slice boundary.
        let mask = mask32::<REVERT>(low, high, _mm256_loadu_si256(ptr.cast()));
        if mask != 0 {
            // 31 - leading_zeros locates the highest set bit in this u32,
            // which identifies the last selected byte lane.
            Some((31 - mask.leading_zeros()) as usize)
        } else {
            None
        }
    }
}

// Bit i selects byte i: members for find, nonmembers when REVERT is true.
// Each high-table entry must be zero or contain exactly one group flag.
#[inline(always)]
unsafe fn mask32<const REVERT: bool>(low: __m128i, high: __m128i, bytes: __m256i) -> u32 {
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
        // A nonzero intersection means membership. movemask only consumes
        // the top bit of each byte, not the remaining seven bits.
        let member = _mm256_and_si256(low, high);
        let selected = if REVERT {
            // Zero becomes 0b11111111; nonzero becomes 0b00000000.
            _mm256_cmpeq_epi8(member, _mm256_setzero_si256())
        } else {
            // Each high-table entry selects at most one group, so member is
            // zero or a power of two <= 128. Adding 127 cannot overflow:
            // zero becomes 127 (top bit clear), a member 128..255 (top bit set).
            _mm256_add_epi8(member, _mm256_set1_epi8(0b0111_1111))
        };
        // Pack selected byte i's top bit into scalar bit i; no inversion remains.
        _mm256_movemask_epi8(selected) as u32
    }
}

// Both vector halves classify the same sixteen bytes; bit i selects byte i.
// Each high-table entry must be zero or contain exactly one group flag.
#[inline(always)]
unsafe fn mask16_avx2<const REVERT: bool>(low: __m128i, high: __m128i, bytes: __m128i) -> u16 {
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
        let selected = if REVERT {
            // Zero intersections select nonmembers, producing all-one bytes.
            _mm_cmpeq_epi8(member, _mm_setzero_si128())
        } else {
            // The high table selects at most one group, bounding member by 128.
            // Adding 127 maps zero to 127 and members to 128..255 without
            // overflow. Only the top bit matters to movemask.
            _mm_add_epi8(member, _mm_set1_epi8(0b0111_1111))
        };
        _mm_movemask_epi8(selected) as u16
    }
}

#[cfg(test)]
mod tests {
    use super::{_mm256_setzero_si256, _mm_set1_epi8, _mm_setzero_si128, mask16_avx2, mask32};
    use crate::SimdBackend;

    #[test]
    fn mask32_selects_each_group_flag() {
        if !SimdBackend::Avx2.is_available() {
            return;
        }
        // Exercise every group, including bit 7: 128 + 127 must select the lane.
        for group in 0..8 {
            let flag = 1u8 << group;
            // SAFETY: AVX2 is available; all vector values are initialized.
            let actual = unsafe {
                let table = _mm_set1_epi8(flag as i8);
                mask32::<false>(table, table, _mm256_setzero_si256())
            };
            assert_eq!(actual, u32::MAX, "every lane must select flag {flag:#010b}");
        }
    }

    #[test]
    fn mask16_avx2_selects_each_group_flag() {
        if !SimdBackend::Avx2.is_available() {
            return;
        }
        // The combined lookup must select all eight possible group flags.
        for group in 0..8 {
            let flag = 1u8 << group;
            // SAFETY: AVX2 is available; all vector values are initialized.
            let actual = unsafe {
                let table = _mm_set1_epi8(flag as i8);
                mask16_avx2::<false>(table, table, _mm_setzero_si128())
            };
            assert_eq!(actual, u16::MAX, "every lane must select flag {flag:#010b}");
        }
    }
}
