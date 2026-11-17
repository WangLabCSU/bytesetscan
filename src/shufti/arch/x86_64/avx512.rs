#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// Requires avx512f,avx512bw, valid table arrays, any slice length.
#[target_feature(enable = "avx512f,avx512bw")]
pub(in super::super) fn shufti_avx512<const REVERT: bool>(
    low: &[u8; 16],
    high: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // SAFETY: array loads stay within their initialized storage; dispatch
    // validates CPU support; short inputs are handled before full input loads.
    unsafe {
        search512::<REVERT>(
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
pub(super) unsafe fn search512<const REVERT: bool>(
    low: __m128i,
    high: __m128i,
    haystack: &[u8],
) -> usize {
    // SAFETY: the entry point establishes AVX-512F/BW support. Masked loads
    // access only enabled bytes; all full loads below stay inside haystack.
    unsafe {
        let base = haystack.as_ptr();
        let len = haystack.len();
        // Masked loads let AVX-512 process a short slice without touching
        // adjacent bytes. Masking the classification result is also necessary:
        // inactive lanes contain zero, which may itself belong to the needle.
        if len <= 64 {
            let valid = u64::MAX.checked_shr((64 - len) as u32).unwrap_or(0);
            // Masked 64-byte load: bit i of valid enables memory byte i;
            // disabled lanes are zero-filled. No 64-byte alignment is required.
            // Classification is masked again below so zero-filled lanes cannot
            // be reported even if byte value zero matches the needle.
            let mut mask = mask64(low, high, _mm512_maskz_loadu_epi8(valid, base.cast()));
            if !REVERT {
                // Convert the classifier's nonmember bits into member bits. This
                // changes the predicate only; forward/reverse bit ordering is unchanged.
                mask = !mask;
            }
            // Discard positions outside the actual input, including any padding
            // whose zero byte happened to satisfy the selected predicate.
            mask &= valid;
            return if mask != 0 {
                // The lowest set bit identifies the first selected lane.
                mask.trailing_zeros() as usize
            } else {
                len
            };
        }
        let mut start = 0;
        // The low six address bits give the remainder modulo 64. When it is
        // nonzero, 64 - remainder is the distance to the next 64-byte boundary.
        // Scan that prefix first; an aligned base starts the main loop at 0.
        let misalignment = base as usize & 0b00111111;
        if misalignment != 0 {
            start = 64 - misalignment;
            let valid = u64::MAX >> (64 - start);
            // Masked 64-byte load: bit i of valid enables memory byte i;
            // disabled lanes are zero-filled. No 64-byte alignment is required.
            // Classification is masked again below so zero-filled lanes cannot
            // be reported even if byte value zero matches the needle.
            let mut mask = mask64(low, high, _mm512_maskz_loadu_epi8(valid, base.cast()));
            if !REVERT {
                // Convert the classifier's nonmember bits into member bits. This
                // changes the predicate only; forward/reverse bit ordering is unchanged.
                mask = !mask;
            }
            // Discard positions outside the actual input, including any padding
            // whose zero byte happened to satisfy the selected predicate.
            mask &= valid;
            if mask != 0 {
                // The lowest set bit identifies the first selected lane.
                return mask.trailing_zeros() as usize;
            }
        }
        // Subtract the end address's remainder modulo 64 from len. Then
        // base + aligned_end is the last 64-byte boundary at or before its end.
        let aligned_end = len - (base.add(len) as usize & 0b00111111);
        while start < aligned_end {
            // Load all 64 bytes from the 64-byte-aligned interior position;
            // the loop bounds also guarantee a complete vector remains.
            let mut mask = mask64(low, high, _mm512_load_si512(base.add(start).cast()));
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
            start += 64;
        }
        if start == len {
            return len;
        }
        let last = len - 64;
        // Load all 64 bytes at this address without an alignment requirement.
        // The full-vector path keeps the entire read inside haystack.
        let mut mask = mask64(low, high, _mm512_loadu_si512(base.add(last).cast()));
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

// Reverse entry point. Requires avx512f,avx512bw any slice length.
#[target_feature(enable = "avx512f,avx512bw")]
pub(in super::super) fn rshufti_avx512<const REVERT: bool>(
    low: &[u8; 16],
    high: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // SAFETY: array loads stay within their initialized storage; dispatch
    // validates CPU support; short inputs are handled before full input loads.
    unsafe {
        rsearch512::<REVERT>(
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
pub(super) unsafe fn rsearch512<const REVERT: bool>(
    low: __m128i,
    high: __m128i,
    haystack: &[u8],
) -> usize {
    // SAFETY: the entry point establishes AVX-512F/BW support. Masked loads
    // touch only enabled bytes; each full load is preceded by a width check.
    unsafe {
        let base = haystack.as_ptr();
        let len = haystack.len();
        if len < 64 {
            let valid = u64::MAX.checked_shr((64 - len) as u32).unwrap_or(0);
            // Masked 64-byte load: bit i of valid enables memory byte i;
            // disabled lanes are zero-filled. No 64-byte alignment is required.
            // Classification is masked again below so zero-filled lanes cannot
            // be reported even if byte value zero matches the needle.
            let mut mask = mask64(low, high, _mm512_maskz_loadu_epi8(valid, base.cast()));
            if !REVERT {
                // Convert the classifier's nonmember bits into member bits. This
                // changes the predicate only; forward/reverse bit ordering is unchanged.
                mask = !mask;
            }
            // Inactive zero-filled lanes must not appear in either search.
            // Discard positions outside the actual input, including any padding
            // whose zero byte happened to satisfy the selected predicate.
            mask &= valid;
            return if mask != 0 {
                // 63 - leading_zeros locates the highest set bit in this u64,
                // which identifies the last selected byte lane.
                (63 - mask.leading_zeros()) as usize
            } else {
                len
            };
        }

        // The end address's remainder modulo 64 is the suffix length after
        // the preceding 64-byte boundary. Scan those bytes before the aligned
        // loop; enabling only suffix lanes keeps the load inside the slice.
        let suffix = base.add(len) as usize & 0b00111111;
        let mut end = len - suffix;
        if suffix != 0 {
            let valid = u64::MAX >> (64 - suffix);
            // Masked 64-byte load: bit i of valid enables memory byte i;
            // disabled lanes are zero-filled. No 64-byte alignment is required.
            // Classification is masked again below so zero-filled lanes cannot
            // be reported even if byte value zero matches the needle.
            let mut mask = mask64(
                low,
                high,
                _mm512_maskz_loadu_epi8(valid, base.add(end).cast()),
            );
            if !REVERT {
                // Convert the classifier's nonmember bits into member bits. This
                // changes the predicate only; forward/reverse bit ordering is unchanged.
                mask = !mask;
            }
            // Discard positions outside the actual input, including any padding
            // whose zero byte happened to satisfy the selected predicate.
            mask &= valid;
            if mask != 0 {
                // 63 - leading_zeros locates the highest set bit in this u64,
                // which identifies the last selected byte lane.
                // Add this block's start to return a position in the original input.
                return end + (63 - mask.leading_zeros()) as usize;
            }
        }
        // The inner mask gives the start address's remainder modulo 64.
        // Subtract it from 64 to get the distance to the next boundary.
        // The outer mask changes 64 to 0 when base is already aligned,
        // so the reverse loop stops at the boundary at or after base.
        let first = (64 - (base as usize & 0b00111111)) & 0b00111111;
        while end > first {
            end -= 64;
            // Load all 64 bytes from the 64-byte-aligned interior position;
            // the loop bounds also guarantee a complete vector remains.
            let mut mask = mask64(low, high, _mm512_load_si512(base.add(end).cast()));
            if !REVERT {
                // Convert the classifier's nonmember bits into member bits. This
                // changes the predicate only; forward/reverse bit ordering is unchanged.
                mask = !mask;
            }
            if mask != 0 {
                // 63 - leading_zeros locates the highest set bit in this u64,
                // which identifies the last selected byte lane.
                // Add this block's start to return a position in the original input.
                return end + (63 - mask.leading_zeros()) as usize;
            }
        }
        if end == 0 {
            return len;
        }
        // len >= 64 guarantees a full prefix; the overlap has already failed.
        // Load all 64 bytes at this address without an alignment requirement.
        // The full-vector path keeps the entire read inside haystack.
        let mut mask = mask64(low, high, _mm512_loadu_si512(base.cast()));
        if !REVERT {
            // Convert the classifier's nonmember bits into member bits. This
            // changes the predicate only; forward/reverse bit ordering is unchanged.
            mask = !mask;
        }
        if mask != 0 {
            // 63 - leading_zeros locates the highest set bit in this u64,
            // which identifies the last selected byte lane.
            (63 - mask.leading_zeros()) as usize
        } else {
            len
        }
    }
}

#[inline(always)]
unsafe fn mask64(low: __m128i, high: __m128i, bytes: __m512i) -> u64 {
    // SAFETY: the caller has established AVX-512F/BW support.
    unsafe {
        // set1_epi8 broadcasts 0b00001111 into every byte lane. AND with this
        // mask extracts each input byte's low nibble as a table index in 0..16.
        let nibble = _mm512_set1_epi8(0b00001111);
        // shuffle_epi8 performs the low-table lookup independently in each
        // 128-bit lane. The following broadcast supplies an identical table
        // to every lane; AND supplies the input bytes' low-nibble indices.
        let low = _mm512_shuffle_epi8(
            // Replicate the complete 16-byte low table across the wide register;
            // this does not merge the shuffle's separate 128-bit lookup domains.
            _mm512_broadcast_i32x4(low),
            _mm512_and_si512(bytes, nibble),
        );
        // Look up the high nibble in the high table. andnot(a, b) means
        // (~a) & b: clear each byte's low four bits BEFORE shifting the
        // 64-bit lanes. This prevents a neighboring byte's low nibble from
        // spilling into this byte; the resulting indices are exactly b >> 4.
        let high = _mm512_shuffle_epi8(
            // Repeat the 16-byte high table in each 128-bit shuffle lane.
            _mm512_broadcast_i32x4(high),
            _mm512_srli_epi64::<4>(_mm512_andnot_si512(nibble, bytes)),
        );
        // AND intersects the low/high group flags. Compare each result
        // with zero: cmpeq_epi8_mask directly produces 64 scalar predicate
        // bits (1 = nonmember), without a separate byte movemask step.
        _mm512_cmpeq_epi8_mask(_mm512_and_si512(low, high), _mm512_setzero_si512())
    }
}
