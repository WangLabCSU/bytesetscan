// Intrinsic reference (128-bit lane boundaries and index-bit zeroing):
// https://doc.rust-lang.org/core/arch/x86_64/fn._mm256_shuffle_epi8.html
// Masked byte loads:
// https://doc.rust-lang.org/core/arch/x86_64/fn._mm512_maskz_loadu_epi8.html

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// The mask classifiers identify nonmembers in blocks of 16, 32, or 64 bytes.
// Each scan handles its initial block, aligned interior blocks, and final
// overlapping block. SSE4.2 shares the SSSE3 scan; VBMI shares the AVX-512 scan.
//
// Scans return a slice index, using haystack.len() when no byte is selected.
// Empty slices return the same sentinel. Reverse traversal never constructs
// a pointer before the slice. REVERT selects nonmembers without changing
// scan direction. Kernel calls are statically dispatched and inlined.
//
// Each mask marks nonmembers; the scan specializes find versus mismatch.
// Callers must establish the instruction set required by each classifier.
// Mask convention: bit i corresponds to the i-th loaded byte; 1 means
// that byte is outside the needle. Find flips this mask, mismatch
// keeps it. Forward selects the lowest set bit; reverse the highest.
// _mm / _mm256 / _mm512 below operate on 16 / 32 / 64 bytes.
// Byte shuffle instructions still use separate 16-byte tables in
// each 128-bit lane, even when the overall register is wider.
trait Kernel {
    // SSSE3 classification; this must not call the AVX2 short-block variant.
    unsafe fn mask16(&self, bytes: __m128i) -> u16;
    // AVX-512F/BW classification returns the native 64-bit predicate mask.
    unsafe fn mask64(&self, bytes: __m512i) -> u64;
    // Full AVX2 block.
    unsafe fn mask32(&self, bytes: __m256i) -> u32;
    // This slot reuses the SSSE3 classifier and does not require AVX2.
    // Truffle's AVX2 scans use mask32 even for short inputs.
    unsafe fn mask16_avx2(&self, bytes: __m128i) -> u16;
}

// Register copies of TruffleTable's two masks. Their byte-sized entries
// can be looked up directly with pshufb.
// The input's top bit selects lower/upper; its low nibble selects an entry,
// and bits 4..6 select the membership bit within that entry.
struct Truffle {
    lower: __m128i,
    upper: __m128i,
}

/// # Safety
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn truffle_avx2<const REVERT: bool>(
    lower: &[u8; 16],
    upper: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // SAFETY: dispatch establishes CPU support; the scan handles short inputs.
    // Both table arrays provide one initialized 16-byte vector.
    unsafe {
        search::<_, REVERT>(
            Truffle {
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                lower: _mm_loadu_si128(lower.as_ptr().cast()),
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                upper: _mm_loadu_si128(upper.as_ptr().cast()),
            },
            haystack,
        )
    }
}

#[inline(always)]
unsafe fn search<K: Kernel, const REVERT: bool>(kernel: K, haystack: &[u8]) -> usize {
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
            let mut mask = kernel.mask32(_mm256_loadu_si256(bytes.as_ptr().cast()));
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
        let mut mask = kernel.mask32(_mm256_loadu_si256(base.cast()));
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
        // Advance to the next vector-aligned address after testing the
        // initial unaligned block. Any overlap contains only rejected bytes.
        let mut start = 32 - (base as usize & 0b00011111);
        let last = len - 32;
        while start < last {
            // Read 32 bytes from the 32-byte-aligned interior address selected
            // by start/end; an aligned load still needs the full vector in bounds.
            let mut mask = kernel.mask32(_mm256_load_si256(base.add(start).cast()));
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
        let mut mask = kernel.mask32(_mm256_loadu_si256(base.add(last).cast()));
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

// Requires ssse3, valid table arrays, any slice length.
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn truffle_ssse3<const REVERT: bool>(
    lower: &[u8; 16],
    upper: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // SAFETY: dispatch establishes CPU support; the scan handles short inputs.
    // Both table arrays provide one initialized 16-byte vector.
    unsafe {
        search128::<_, REVERT>(
            Truffle {
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                lower: _mm_loadu_si128(lower.as_ptr().cast()),
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                upper: _mm_loadu_si128(upper.as_ptr().cast()),
            },
            haystack,
        )
    }
}

// Requires ssse3,sse4.2,popcnt, valid table arrays, any slice length.
#[target_feature(enable = "ssse3,sse4.2,popcnt")]
pub(super) unsafe fn truffle_sse42<const REVERT: bool>(
    lower: &[u8; 16],
    upper: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // SAFETY: dispatch establishes CPU support; the scan handles short inputs.
    // Both table arrays provide one initialized 16-byte vector.
    unsafe {
        search128::<_, REVERT>(
            Truffle {
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                lower: _mm_loadu_si128(lower.as_ptr().cast()),
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                upper: _mm_loadu_si128(upper.as_ptr().cast()),
            },
            haystack,
        )
    }
}

// SSSE3 and SSE4.2 share the initial/aligned/final 16-byte scan.
#[inline(always)]
unsafe fn search128<K: Kernel, const REVERT: bool>(kernel: K, haystack: &[u8]) -> usize {
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
            let mut mask = kernel.mask16(_mm_loadu_si128(bytes.as_ptr().cast()));
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
        let mut mask = kernel.mask16(_mm_loadu_si128(base.cast()));
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
        // Advance to the next vector-aligned address after testing the
        // initial unaligned block. Any overlap contains only rejected bytes.
        let mut start = 16 - (base as usize & 0b00001111);
        let last = len - 16;
        while start < last {
            // Load sixteen bytes from the 16-byte-aligned interior address;
            // the loop excludes positions without a complete input vector.
            let mut mask = kernel.mask16(_mm_load_si128(base.add(start).cast()));
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
        let mut mask = kernel.mask16(_mm_loadu_si128(base.add(last).cast()));
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

// Requires avx512f,avx512bw, valid table arrays, any slice length.
#[target_feature(enable = "avx512f,avx512bw")]
pub(super) unsafe fn truffle_avx512<const REVERT: bool>(
    lower: &[u8; 16],
    upper: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // SAFETY: array loads stay within their initialized storage; dispatch
    // validates CPU support; short inputs are handled before full input loads.
    unsafe {
        search512::<_, REVERT>(
            Truffle {
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                lower: _mm_loadu_si128(lower.as_ptr().cast()),
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                upper: _mm_loadu_si128(upper.as_ptr().cast()),
            },
            haystack,
        )
    }
}

// AVX-512 and VBMI use the same single-byte shuffle classifier. Separate entry
// points allow selecting VBMI without requiring it for the base AVX-512 path.
// Performance still needs hardware validation.
// Requires avx512f,avx512bw,avx512vbmi, valid table arrays, any slice length.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
pub(super) unsafe fn truffle_vbmi<const REVERT: bool>(
    lower: &[u8; 16],
    upper: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // SAFETY: array loads stay within their initialized storage; dispatch
    // validates CPU support; short inputs are handled before full input loads.
    unsafe {
        search512::<_, REVERT>(
            Truffle {
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                lower: _mm_loadu_si128(lower.as_ptr().cast()),
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                upper: _mm_loadu_si128(upper.as_ptr().cast()),
            },
            haystack,
        )
    }
}

#[inline(always)]
unsafe fn search512<K: Kernel, const REVERT: bool>(kernel: K, haystack: &[u8]) -> usize {
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
            let mut mask = kernel.mask64(_mm512_maskz_loadu_epi8(valid, base.cast()));
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
        // Read only up to the first aligned boundary. The main loop then uses
        // aligned vectors; an overlapping final vector covers any suffix.
        let misalignment = base as usize & 0b00111111;
        if misalignment != 0 {
            start = 64 - misalignment;
            let valid = u64::MAX >> (64 - start);
            // Masked 64-byte load: bit i of valid enables memory byte i;
            // disabled lanes are zero-filled. No 64-byte alignment is required.
            // Classification is masked again below so zero-filled lanes cannot
            // be reported even if byte value zero matches the needle.
            let mut mask = kernel.mask64(_mm512_maskz_loadu_epi8(valid, base.cast()));
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
            let mut mask = kernel.mask64(_mm512_load_si512(base.add(start).cast()));
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
        let mut mask = kernel.mask64(_mm512_loadu_si512(base.add(last).cast()));
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
pub(super) unsafe fn rtruffle_avx2<const REVERT: bool>(
    lower: &[u8; 16],
    upper: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // SAFETY: dispatch establishes CPU support; the scan handles short inputs.
    // Both table arrays provide one initialized 16-byte vector.
    unsafe {
        rsearch::<_, REVERT>(
            Truffle {
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                lower: _mm_loadu_si128(lower.as_ptr().cast()),
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                upper: _mm_loadu_si128(upper.as_ptr().cast()),
            },
            haystack,
        )
    }
}

// Reverse counterpart to the forward scan: unaligned suffix, aligned interior,
// then an overlapping prefix. The highest set bit identifies the last match.
#[inline(always)]
unsafe fn rsearch<K: Kernel, const REVERT: bool>(kernel: K, haystack: &[u8]) -> usize {
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
            let mut mask = kernel.mask32(_mm256_loadu_si256(bytes.as_ptr().cast()));
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
        let mut mask = kernel.mask32(_mm256_loadu_si256(base.add(last).cast()));
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
        // Round the exclusive end down to a vector boundary. The initial
        // suffix scan has already ruled out any bytes skipped by rounding.
        let mut end = len - (base.add(len) as usize & 0b00011111);
        // The source compares buf_end with buf + 32 before decrementing.
        while end > 32 {
            end -= 32;
            // Read 32 bytes from the 32-byte-aligned interior address selected
            // by start/end; an aligned load still needs the full vector in bounds.
            let mut mask = kernel.mask32(_mm256_load_si256(base.add(end).cast()));
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
        let mut mask = kernel.mask32(_mm256_loadu_si256(base.cast()));
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

// Reverse entry point. Requires ssse3 any slice length.
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn rtruffle_ssse3<const REVERT: bool>(
    lower: &[u8; 16],
    upper: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // SAFETY: dispatch establishes CPU support; the scan handles short inputs.
    // Both table arrays provide one initialized 16-byte vector.
    unsafe {
        rsearch128::<_, REVERT>(
            Truffle {
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                lower: _mm_loadu_si128(lower.as_ptr().cast()),
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                upper: _mm_loadu_si128(upper.as_ptr().cast()),
            },
            haystack,
        )
    }
}

// Reverse entry point. Requires ssse3,sse4.2,popcnt any slice length.
#[target_feature(enable = "ssse3,sse4.2,popcnt")]
pub(super) unsafe fn rtruffle_sse42<const REVERT: bool>(
    lower: &[u8; 16],
    upper: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // SAFETY: dispatch establishes CPU support; the scan handles short inputs.
    // Both table arrays provide one initialized 16-byte vector.
    unsafe {
        rsearch128::<_, REVERT>(
            Truffle {
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                lower: _mm_loadu_si128(lower.as_ptr().cast()),
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                upper: _mm_loadu_si128(upper.as_ptr().cast()),
            },
            haystack,
        )
    }
}

// Reverse counterpart to the forward scan: unaligned suffix, aligned interior,
// then an overlapping prefix. The highest set bit identifies the last match.
#[inline(always)]
unsafe fn rsearch128<K: Kernel, const REVERT: bool>(kernel: K, haystack: &[u8]) -> usize {
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
            let mut mask = kernel.mask16(_mm_loadu_si128(bytes.as_ptr().cast()));
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
        let mut mask = kernel.mask16(_mm_loadu_si128(base.add(last).cast()));
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
        // Round the exclusive end down to a vector boundary. The initial
        // suffix scan has already ruled out any bytes skipped by rounding.
        let mut end = len - (base.add(len) as usize & 0b00001111);
        // The source compares buf_end with buf + 16 before decrementing.
        while end > 16 {
            end -= 16;
            // Load sixteen bytes from the 16-byte-aligned interior address;
            // the loop excludes positions without a complete input vector.
            let mut mask = kernel.mask16(_mm_load_si128(base.add(end).cast()));
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
        let mut mask = kernel.mask16(_mm_loadu_si128(base.cast()));
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

// Reverse entry point. Requires avx512f,avx512bw any slice length.
#[target_feature(enable = "avx512f,avx512bw")]
pub(super) unsafe fn rtruffle_avx512<const REVERT: bool>(
    lower: &[u8; 16],
    upper: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // SAFETY: array loads stay within their initialized storage; dispatch
    // validates CPU support; short inputs are handled before full input loads.
    unsafe {
        rsearch512::<_, REVERT>(
            Truffle {
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                lower: _mm_loadu_si128(lower.as_ptr().cast()),
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                upper: _mm_loadu_si128(upper.as_ptr().cast()),
            },
            haystack,
        )
    }
}

// Reverse entry point. Requires avx512f,avx512bw,avx512vbmi any slice length.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
pub(super) unsafe fn rtruffle_vbmi<const REVERT: bool>(
    lower: &[u8; 16],
    upper: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // SAFETY: array loads stay within their initialized storage; dispatch
    // validates CPU support; short inputs are handled before full input loads.
    unsafe {
        rsearch512::<_, REVERT>(
            Truffle {
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                lower: _mm_loadu_si128(lower.as_ptr().cast()),
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                upper: _mm_loadu_si128(upper.as_ptr().cast()),
            },
            haystack,
        )
    }
}

#[inline(always)]
unsafe fn rsearch512<K: Kernel, const REVERT: bool>(kernel: K, haystack: &[u8]) -> usize {
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
            let mut mask = kernel.mask64(_mm512_maskz_loadu_epi8(valid, base.cast()));
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
        let mut mask = kernel.mask64(_mm512_loadu_si512(base.add(last).cast()));
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
        // Round the exclusive end down to a vector boundary. The initial
        // suffix scan has already ruled out any bytes skipped by rounding.
        let mut end = len - (base.add(len) as usize & 0b00111111);
        while end > 64 {
            end -= 64;
            // Load all 64 bytes from the 64-byte-aligned interior position;
            // the loop bounds also guarantee a complete vector remains.
            let mut mask = kernel.mask64(_mm512_load_si512(base.add(end).cast()));
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
        let mut mask = kernel.mask64(_mm512_loadu_si512(base.cast()));
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

impl Kernel for Truffle {
    #[inline(always)]
    unsafe fn mask16(&self, bytes: __m128i) -> u16 {
        // SAFETY: this mask16_avx2 implementation requires only SSSE3.
        unsafe { self.mask16_avx2(bytes) }
    }
    #[inline(always)]
    unsafe fn mask64(&self, bytes: __m512i) -> u64 {
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
            let lower = _mm512_shuffle_epi8(_mm512_broadcast_i32x4(self.lower), bytes);
            // XOR with 0b10000000 toggles only the index's top bit, preserving the
            // low-nibble table index. This shuffle now contributes only for
            // bytes at least 128; exactly one of lower/upper can be nonzero.
            let upper = _mm512_shuffle_epi8(
                _mm512_broadcast_i32x4(self.upper),
                _mm512_xor_si512(bytes, top),
            );
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

    #[inline(always)]
    unsafe fn mask32(&self, bytes: __m256i) -> u32 {
        // SAFETY: called only from an AVX2 entry point. pshufb zeros a lane
        // when bit 7 of the index is set; flipping that bit selects the half.
        unsafe {
            // set1_epi8(0b10000000u8 as i8) repeats the bit pattern 0b10000000 in every byte.
            // This is the shuffle index's zeroing bit, not a signed comparison.
            let top = _mm256_set1_epi8(0b1000_0000u8 as i8);
            // Repeating these eight byte values turns the high nibble into
            // one bit. For 0b11000001, the upper table supplies entry 1 and bit 4 is
            // tested; the lower-table shuffle is suppressed by the top bit.
            // set1_epi64[x] repeats the 64-bit constant; in x86 byte order it
            // contains [0b00000001, 0b00000010, 0b00000100, 0b00001000, 0b00010000, 0b00100000,
            // 0b01000000, 0b10000000]. Each 16-byte shuffle lane
            // therefore has two copies. Index h returns 1 << (h & 0b00000111).
            let bits = _mm256_set1_epi64x(
                0b10000000_01000000_00100000_00010000_00001000_00000100_00000010_00000001u64 as i64,
            );
            // PSHUFB selects table[byte & 0b00001111] unless index bit 7 is set,
            // in which case it returns zero. Thus this lookup contributes only
            // for bytes below 128. On wide vectors, broadcast repeats the full
            // 16-byte table in each independent 128-bit shuffle lane.
            let lower = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(self.lower), bytes);
            // XOR with 0b10000000 toggles only the index's top bit, preserving the
            // low-nibble table index. This shuffle now contributes only for
            // bytes at least 128; exactly one of lower/upper can be nonzero.
            let upper = _mm256_shuffle_epi8(
                _mm256_broadcastsi128_si256(self.upper),
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

    #[inline(always)]
    unsafe fn mask16_avx2(&self, bytes: __m128i) -> u16 {
        // SAFETY: the caller has established SSSE3 support. The SSSE3 and
        // SSE4.2 scans share this classifier for a single 16-byte block.
        unsafe {
            // set1_epi8(0b10000000u8 as i8) repeats the bit pattern 0b10000000 in every byte.
            // This is the shuffle index's zeroing bit, not a signed comparison.
            let top = _mm_set1_epi8(0b10000000u8 as i8);
            // set1_epi64[x] repeats the 64-bit constant; in x86 byte order it
            // contains [0b00000001, 0b00000010, 0b00000100, 0b00001000, 0b00010000, 0b00100000,
            // 0b01000000, 0b10000000]. Each 16-byte shuffle lane
            // therefore has two copies. Index h returns 1 << (h & 0b00000111).
            let bits = _mm_set1_epi64x(
                0b10000000_01000000_00100000_00010000_00001000_00000100_00000010_00000001u64 as i64,
            );
            // PSHUFB selects table[byte & 0b00001111] unless index bit 7 is set,
            // in which case it returns zero. Thus this lookup contributes only
            // for bytes below 128. On wide vectors, broadcast repeats the full
            // 16-byte table in each independent 128-bit shuffle lane.
            let lower = _mm_shuffle_epi8(self.lower, bytes);
            // XOR with 0b10000000 toggles only the index's top bit, preserving the
            // low-nibble table index. This shuffle now contributes only for
            // bytes at least 128; exactly one of lower/upper can be nonzero.
            let upper = _mm_shuffle_epi8(self.upper, _mm_xor_si128(bytes, top));
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
}
