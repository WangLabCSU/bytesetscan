#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// Requires avx512f,avx512bw, valid table arrays, any slice length.
#[target_feature(enable = "avx512f,avx512bw")]
pub(in super::super) unsafe fn truffle_avx512<const REVERT: bool>(
    lower: &[u8; 16],
    upper: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // SAFETY: array loads stay within their initialized storage; dispatch
    // validates CPU support; short inputs are handled before full input loads.
    unsafe {
        search512::<REVERT>(
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
pub(super) unsafe fn search512<const REVERT: bool>(
    lower: __m128i,
    upper: __m128i,
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
            let mut mask = mask64(lower, upper, _mm512_maskz_loadu_epi8(valid, base.cast()));
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
            let mut mask = mask64(lower, upper, _mm512_maskz_loadu_epi8(valid, base.cast()));
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
        let last = len - 64;
        while start < last {
            // Load all 64 bytes from the 64-byte-aligned interior position;
            // the loop bounds also guarantee a complete vector remains.
            let mut mask = mask64(lower, upper, _mm512_load_si512(base.add(start).cast()));
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
        // Load all 64 bytes at this address without an alignment requirement.
        // The full-vector path keeps the entire read inside haystack.
        let mut mask = mask64(lower, upper, _mm512_loadu_si512(base.add(last).cast()));
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
pub(in super::super) unsafe fn rtruffle_avx512<const REVERT: bool>(
    lower: &[u8; 16],
    upper: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // SAFETY: array loads stay within their initialized storage; dispatch
    // validates CPU support; short inputs are handled before full input loads.
    unsafe {
        rsearch512::<REVERT>(
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
pub(super) unsafe fn rsearch512<const REVERT: bool>(
    lower: __m128i,
    upper: __m128i,
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
            let mut mask = mask64(lower, upper, _mm512_maskz_loadu_epi8(valid, base.cast()));
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

        // rtruffleExec checks a full unaligned suffix before rounding the end
        // down. Unlike rshuftiExec, it does not peel a masked partial suffix.
        let last = len - 64;
        // Load all 64 bytes at this address without an alignment requirement.
        // The full-vector path keeps the entire read inside haystack.
        let mut mask = mask64(lower, upper, _mm512_loadu_si512(base.add(last).cast()));
        if !REVERT {
            // Convert the classifier's nonmember bits into member bits. This
            // changes the predicate only; forward/reverse bit ordering is unchanged.
            mask = !mask;
        }
        if mask != 0 {
            // 63 - leading_zeros locates the highest set bit in this u64,
            // which identifies the last selected byte lane.
            // Add this block's start to return a position in the original input.
            return last + (63 - mask.leading_zeros()) as usize;
        }
        // The end address's low six bits give its remainder modulo 64.
        // Subtract that remainder from len to round base + end down to a
        // 64-byte boundary. The initial suffix scan covered the skipped bytes.
        let mut end = len - (base.add(len) as usize & 0b00111111);
        while end > 64 {
            end -= 64;
            // Load all 64 bytes from the 64-byte-aligned interior position;
            // the loop bounds also guarantee a complete vector remains.
            let mut mask = mask64(lower, upper, _mm512_load_si512(base.add(end).cast()));
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
        // The final full prefix is also read when it overlaps an earlier block.
        // Load all 64 bytes at this address without an alignment requirement.
        // The full-vector path keeps the entire read inside haystack.
        let mut mask = mask64(lower, upper, _mm512_loadu_si512(base.cast()));
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
unsafe fn mask64(lower: __m128i, upper: __m128i, bytes: __m512i) -> u64 {
    // SAFETY: the caller has established AVX-512F/BW support.
    unsafe {
        // set1_epi8(0b10000000u8 as i8) repeats the bit pattern 0b10000000 in every byte.
        // This is the shuffle index's zeroing bit, not a signed comparison.
        let top = _mm512_set1_epi8(0b10000000u8 as i8);
        // set1_epi64[x] repeats the 64-bit constant; in x86 byte order it
        // contains [0b00000001, 0b00000010, 0b00000100, 0b00001000, 0b00010000, 0b00100000,
        // 0b01000000, 0b10000000]. Each 16-byte shuffle lane
        // therefore has two copies. Index h returns 1 << (h & 0b00000111).
        let bits = _mm512_set1_epi64(
            0b10000000_01000000_00100000_00010000_00001000_00000100_00000010_00000001u64 as i64,
        );
        // PSHUFB selects table[byte & 0b00001111] unless index bit 7 is set,
        // in which case it returns zero. Thus this lookup contributes only
        // for bytes below 128. On wide vectors, broadcast repeats the full
        // 16-byte table in each independent 128-bit shuffle lane.
        let lower = _mm512_shuffle_epi8(_mm512_broadcast_i32x4(lower), bytes);
        // XOR with 0b10000000 toggles only the index's top bit, preserving the
        // low-nibble table index. This shuffle now contributes only for
        // bytes at least 128; exactly one of lower/upper can be nonzero.
        let upper =
            _mm512_shuffle_epi8(_mm512_broadcast_i32x4(upper), _mm512_xor_si512(bytes, top));
        // srli_epi64 shifts 64-bit lanes, not individual bytes. Its low
        // four output bits still equal this byte's high nibble; higher bits
        // may come from the next byte. andnot(top, shifted) clears bit 7
        // to prevent accidental PSHUFB zeroing; PSHUFB ignores bits 4..6.
        let index = _mm512_andnot_si512(top, _mm512_srli_epi64::<4>(bytes));
        // Shuffle the repeated powers of two by the high nibble. This
        // produces the membership bit 1 << ((byte >> 4) & 0b00000111) in every lane.
        // For 0b11000001 the upper table supplies entry 1 and selected is 0b00010000.
        let selected = _mm512_shuffle_epi8(bits, index);
        // OR joins the active table flags, AND tests the selected bit,
        // and equality with zero produces one nonmember bit per byte.
        // The AVX-512 compare writes the 64-bit predicate mask directly.
        _mm512_cmpeq_epi8_mask(
            _mm512_and_si512(_mm512_or_si512(lower, upper), selected),
            _mm512_setzero_si512(),
        )
    }
}
