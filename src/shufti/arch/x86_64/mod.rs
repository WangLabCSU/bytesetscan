mod avx2;
mod avx512;
mod ssse3;

pub(super) use self::{
    avx2::{rshufti_avx2, shufti_avx2},
    avx512::{rshufti_avx512, shufti_avx512},
    ssse3::{rshufti_ssse3, shufti_ssse3},
};

// The mask classifiers identify nonmembers in blocks of 16, 32, or 64 bytes.
// Each scan handles its initial block, aligned interior blocks, and final
// overlapping block. SSE4.2 dispatches to SSSE3; VBMI dispatches to AVX-512.
//
// Scans return a slice index, using haystack.len() when no byte is selected.
// Empty slices return the same sentinel. Reverse traversal never constructs
// a pointer before the slice. REVERT selects nonmembers without changing
// scan direction. Mask functions are called directly and inlined.
//
// Each mask marks nonmembers; the scan specializes find versus mismatch.
// Callers must establish the instruction set required by each classifier.
// Mask convention: bit i corresponds to the i-th loaded byte; 1 means
// that byte is outside the needle. Find flips this mask, mismatch
// keeps it. Forward selects the lowest set bit; reverse the highest.
// Intrinsics prefixed _mm / _mm256 / _mm512 operate on 16 / 32 / 64 bytes.
// Byte shuffle instructions still use separate 16-byte tables in
// each 128-bit lane, even when the overall register is wider.
