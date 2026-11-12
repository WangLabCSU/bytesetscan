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
    // Full AVX2 block.
    unsafe fn mask32(&self, bytes: __m256i) -> u32;
    // AVX-512F/BW classification returns the native 64-bit predicate mask.
    unsafe fn mask64(&self, bytes: __m512i) -> u64;
    // Short AVX2 block, using a 16-byte load.
    unsafe fn mask16_avx2(&self, bytes: __m128i) -> u16;
}

// Register copies of the two byte-sized tables. Each bit identifies a shared
// row group, so membership requires a common bit in both nibble lookups.
struct Shufti {
    low: __m128i,
    high: __m128i,
}

/// # Safety
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn shufti_avx2<const REVERT: bool>(
    low: &[u8; 16],
    high: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // Use scalar matching when the input cannot fill a 16-byte vector.
    if haystack.len() < 16 {
        return super::search_fallback::<REVERT>(low, high, haystack);
    }
    // SAFETY: each array supplies exactly one initialized 16-byte vector;
    // dispatch establishes CPU support and the short-input check above
    // establishes the minimum length needed by this scan.
    unsafe {
        search::<_, REVERT>(
            Shufti {
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                low: _mm_loadu_si128(low.as_ptr().cast()),
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                high: _mm_loadu_si128(high.as_ptr().cast()),
            },
            haystack,
        )
    }
}

#[inline(always)]
unsafe fn search<K: Kernel, const REVERT: bool>(kernel: K, haystack: &[u8]) -> usize {
    // SAFETY: the entry point establishes AVX2 support and len >= 16.
    // The short path uses two in-bounds 16-byte loads.
    // Aligned interior loads and the final overlapping vector stay in haystack.
    unsafe {
        let base = haystack.as_ptr();
        let len = haystack.len();
        // shuftiFwdShort / shuftiRevShort use two overlapping 16-byte reads.
        if len <= 32 {
            // Load sixteen consecutive bytes without assuming 16-byte alignment.
            // The short path or scan bounds guarantee all sixteen are readable.
            let mut mask = kernel.mask16_avx2(_mm_loadu_si128(base.cast()));
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
            let mut mask = kernel.mask16_avx2(_mm_loadu_si128(base.add(last).cast()));
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
pub(super) unsafe fn shufti_ssse3<const REVERT: bool>(
    low: &[u8; 16],
    high: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // Use scalar matching when the input cannot fill a 16-byte vector.
    if haystack.len() < 16 {
        return super::search_fallback::<REVERT>(low, high, haystack);
    }
    // SAFETY: array loads stay within their initialized storage; dispatch
    // validates CPU support; short inputs are handled before full input loads.
    unsafe {
        search128::<_, REVERT>(
            Shufti {
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                low: _mm_loadu_si128(low.as_ptr().cast()),
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                high: _mm_loadu_si128(high.as_ptr().cast()),
            },
            haystack,
        )
    }
}

// Requires ssse3,sse4.2,popcnt, valid table arrays, any slice length.
#[target_feature(enable = "ssse3,sse4.2,popcnt")]
pub(super) unsafe fn shufti_sse42<const REVERT: bool>(
    low: &[u8; 16],
    high: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // Use scalar matching when the input cannot fill a 16-byte vector.
    if haystack.len() < 16 {
        return super::search_fallback::<REVERT>(low, high, haystack);
    }
    // SAFETY: array loads stay within their initialized storage; dispatch
    // validates CPU support; short inputs are handled before full input loads.
    unsafe {
        search128::<_, REVERT>(
            Shufti {
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                low: _mm_loadu_si128(low.as_ptr().cast()),
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                high: _mm_loadu_si128(high.as_ptr().cast()),
            },
            haystack,
        )
    }
}

// SSSE3 and SSE4.2 share the initial/aligned/final 16-byte scan.
#[inline(always)]
unsafe fn search128<K: Kernel, const REVERT: bool>(kernel: K, haystack: &[u8]) -> usize {
    // SAFETY: the entry point establishes SSSE3 support and len >= 16.
    // The short path is handled by the entry point.
    // Aligned interior loads and the final overlapping vector stay in haystack.
    unsafe {
        let base = haystack.as_ptr();
        let len = haystack.len();
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
pub(super) unsafe fn shufti_avx512<const REVERT: bool>(
    low: &[u8; 16],
    high: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // SAFETY: array loads stay within their initialized storage; dispatch
    // validates CPU support; short inputs are handled before full input loads.
    unsafe {
        search512::<_, REVERT>(
            Shufti {
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                low: _mm_loadu_si128(low.as_ptr().cast()),
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                high: _mm_loadu_si128(high.as_ptr().cast()),
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
pub(super) unsafe fn shufti_vbmi<const REVERT: bool>(
    low: &[u8; 16],
    high: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // SAFETY: array loads stay within their initialized storage; dispatch
    // validates CPU support; short inputs are handled before full input loads.
    unsafe {
        search512::<_, REVERT>(
            Shufti {
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                low: _mm_loadu_si128(low.as_ptr().cast()),
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                high: _mm_loadu_si128(high.as_ptr().cast()),
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
        let aligned_end = len - (base.add(len) as usize & 0b00111111);
        while start < aligned_end {
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
        if start == len {
            return len;
        }
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
pub(super) unsafe fn rshufti_avx2<const REVERT: bool>(
    low: &[u8; 16],
    high: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // Use scalar matching when the input cannot fill a 16-byte vector.
    if haystack.len() < 16 {
        return super::rsearch_fallback::<REVERT>(low, high, haystack);
    }
    // SAFETY: each array supplies exactly one initialized 16-byte vector;
    // dispatch establishes CPU support and the short-input check above
    // establishes the minimum length needed by this scan.
    unsafe {
        rsearch::<_, REVERT>(
            Shufti {
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                low: _mm_loadu_si128(low.as_ptr().cast()),
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                high: _mm_loadu_si128(high.as_ptr().cast()),
            },
            haystack,
        )
    }
}

// Reverse counterpart to the forward scan: unaligned suffix, aligned interior,
// then an overlapping prefix. The highest set bit identifies the last match.
#[inline(always)]
unsafe fn rsearch<K: Kernel, const REVERT: bool>(kernel: K, haystack: &[u8]) -> usize {
    // SAFETY: the entry point establishes AVX2 support and len >= 16.
    // The short path uses two in-bounds 16-byte loads.
    // Aligned interior loads and the final overlapping vector stay in haystack.
    unsafe {
        let base = haystack.as_ptr();
        let len = haystack.len();
        // shuftiFwdShort / shuftiRevShort use two overlapping 16-byte reads.
        if len <= 32 {
            let last = len - 16;
            // Load sixteen consecutive bytes without assuming 16-byte alignment.
            // The short path or scan bounds guarantee all sixteen are readable.
            let mut mask = kernel.mask16_avx2(_mm_loadu_si128(base.add(last).cast()));
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
            let mut mask = kernel.mask16_avx2(_mm_loadu_si128(base.cast()));
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
pub(super) unsafe fn rshufti_ssse3<const REVERT: bool>(
    low: &[u8; 16],
    high: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // Use scalar matching when the input cannot fill a 16-byte vector.
    if haystack.len() < 16 {
        return super::rsearch_fallback::<REVERT>(low, high, haystack);
    }
    // SAFETY: array loads stay within their initialized storage; dispatch
    // validates CPU support; short inputs are handled before full input loads.
    unsafe {
        rsearch128::<_, REVERT>(
            Shufti {
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                low: _mm_loadu_si128(low.as_ptr().cast()),
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                high: _mm_loadu_si128(high.as_ptr().cast()),
            },
            haystack,
        )
    }
}

// Reverse entry point. Requires ssse3,sse4.2,popcnt any slice length.
#[target_feature(enable = "ssse3,sse4.2,popcnt")]
pub(super) unsafe fn rshufti_sse42<const REVERT: bool>(
    low: &[u8; 16],
    high: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // Use scalar matching when the input cannot fill a 16-byte vector.
    if haystack.len() < 16 {
        return super::rsearch_fallback::<REVERT>(low, high, haystack);
    }
    // SAFETY: array loads stay within their initialized storage; dispatch
    // validates CPU support; short inputs are handled before full input loads.
    unsafe {
        rsearch128::<_, REVERT>(
            Shufti {
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                low: _mm_loadu_si128(low.as_ptr().cast()),
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                high: _mm_loadu_si128(high.as_ptr().cast()),
            },
            haystack,
        )
    }
}

// Reverse counterpart to the forward scan: unaligned suffix, aligned interior,
// then an overlapping prefix. The highest set bit identifies the last match.
#[inline(always)]
unsafe fn rsearch128<K: Kernel, const REVERT: bool>(kernel: K, haystack: &[u8]) -> usize {
    // SAFETY: the entry point establishes SSSE3 support and len >= 16.
    // The short path is handled by the entry point.
    // Aligned interior loads and the final overlapping vector stay in haystack.
    unsafe {
        let base = haystack.as_ptr();
        let len = haystack.len();
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
pub(super) unsafe fn rshufti_avx512<const REVERT: bool>(
    low: &[u8; 16],
    high: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // SAFETY: array loads stay within their initialized storage; dispatch
    // validates CPU support; short inputs are handled before full input loads.
    unsafe {
        rsearch512::<_, REVERT>(
            Shufti {
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                low: _mm_loadu_si128(low.as_ptr().cast()),
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                high: _mm_loadu_si128(high.as_ptr().cast()),
            },
            haystack,
        )
    }
}

// Reverse entry point. Requires avx512f,avx512bw,avx512vbmi any slice length.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
pub(super) unsafe fn rshufti_vbmi<const REVERT: bool>(
    low: &[u8; 16],
    high: &[u8; 16],
    haystack: &[u8],
) -> usize {
    // SAFETY: array loads stay within their initialized storage; dispatch
    // validates CPU support; short inputs are handled before full input loads.
    unsafe {
        rsearch512::<_, REVERT>(
            Shufti {
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                low: _mm_loadu_si128(low.as_ptr().cast()),
                // Load the 16 initialized table entries into one register; the
                // array has byte alignment, so use the unaligned load intrinsic.
                high: _mm_loadu_si128(high.as_ptr().cast()),
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

        // Peel the unaligned suffix before scanning complete aligned vectors.
        // Keeping only the suffix lanes avoids reading beyond the slice.
        let suffix = base.add(len) as usize & 0b00111111;
        let mut end = len - suffix;
        if suffix != 0 {
            let valid = u64::MAX >> (64 - suffix);
            // Masked 64-byte load: bit i of valid enables memory byte i;
            // disabled lanes are zero-filled. No 64-byte alignment is required.
            // Classification is masked again below so zero-filled lanes cannot
            // be reported even if byte value zero matches the needle.
            let mut mask = kernel.mask64(_mm512_maskz_loadu_epi8(valid, base.add(end).cast()));
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
        // Stop at the first 64-byte boundary at or after the slice start.
        let first = (64 - (base as usize & 0b00111111)) & 0b00111111;
        while end > first {
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
        if end == 0 {
            return len;
        }
        // len >= 64 guarantees a full prefix; the overlap has already failed.
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

impl Kernel for Shufti {
    #[inline(always)]
    unsafe fn mask16(&self, bytes: __m128i) -> u16 {
        // SAFETY: the caller has established SSSE3 support.
        unsafe {
            // set1_epi8 broadcasts 0b00001111 into every byte lane. AND with this
            // mask extracts each input byte's low nibble as a table index in 0..16.
            let nibble = _mm_set1_epi8(0b00001111);
            // and_si128 extracts sixteen independent low-nibble indices.
            // shuffle_epi8 (PSHUFB) selects self.low[index] for each byte. Its
            // index bit 7 would zero a lane, but the 0b00001111 mask keeps it clear.
            let low = _mm_shuffle_epi8(self.low, _mm_and_si128(bytes, nibble));
            // Look up the high nibble in the high table. andnot(a, b) means
            // (~a) & b: clear each byte's low four bits BEFORE shifting the
            // 64-bit lanes. This prevents a neighboring byte's low nibble from
            // spilling into this byte; the resulting indices are exactly b >> 4.
            let high = _mm_shuffle_epi8(
                self.high,
                _mm_srli_epi64::<4>(_mm_andnot_si128(nibble, bytes)),
            );
            // Compare the intersected flags with a zero vector: 0b11111111 marks
            // a nonmember and 0b00000000 a member. movemask packs the sixteen byte
            // sign bits into the low sixteen bits of its i32 result; cast to u16.
            _mm_movemask_epi8(_mm_cmpeq_epi8(
                _mm_and_si128(low, high),
                _mm_setzero_si128(),
            )) as u16
        }
    }
    #[inline(always)]
    unsafe fn mask64(&self, bytes: __m512i) -> u64 {
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
                _mm512_broadcast_i32x4(self.low),
                _mm512_and_si512(bytes, nibble),
            );
            // Look up the high nibble in the high table. andnot(a, b) means
            // (~a) & b: clear each byte's low four bits BEFORE shifting the
            // 64-bit lanes. This prevents a neighboring byte's low nibble from
            // spilling into this byte; the resulting indices are exactly b >> 4.
            let high = _mm512_shuffle_epi8(
                // Repeat the 16-byte high table in each 128-bit shuffle lane.
                _mm512_broadcast_i32x4(self.high),
                _mm512_srli_epi64::<4>(_mm512_andnot_si512(nibble, bytes)),
            );
            // AND intersects the low/high group flags. Compare each result
            // with zero: cmpeq_epi8_mask directly produces 64 scalar predicate
            // bits (1 = nonmember), without a separate byte movemask step.
            _mm512_cmpeq_epi8_mask(_mm512_and_si512(low, high), _mm512_setzero_si512())
        }
    }

    #[inline(always)]
    unsafe fn mask32(&self, bytes: __m256i) -> u32 {
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
                _mm256_broadcastsi128_si256(self.low),
                _mm256_and_si256(bytes, nibble),
            );
            // Look up the high nibble in the high table. andnot(a, b) means
            // (~a) & b: clear each byte's low four bits BEFORE shifting the
            // 64-bit lanes. This prevents a neighboring byte's low nibble from
            // spilling into this byte; the resulting indices are exactly b >> 4.
            let high = _mm256_shuffle_epi8(
                // Repeat the 16-byte high table in each 128-bit shuffle lane.
                _mm256_broadcastsi128_si256(self.high),
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
    unsafe fn mask16_avx2(&self, bytes: __m128i) -> u16 {
        // SAFETY: called only from an AVX2 entry point. Pack both nibble
        // lookups into one AVX2 shuffle.
        unsafe {
            // set_m128i takes (high_half, low_half): lanes 0..16 hold the low
            // table, lanes 16..32 the high table. Two different lookups can now
            // run in one AVX2 shuffle, still respecting its 128-bit lane boundary.
            let table = _mm256_set_m128i(self.high, self.low);
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
}
