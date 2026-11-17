mod avx2;
mod avx512;
mod ssse3;

pub(super) use self::{
    avx2::{rtruffle_avx2, truffle_avx2},
    avx512::{rtruffle_avx512, truffle_avx512},
    ssse3::{rtruffle_ssse3, truffle_ssse3},
};

// The mask classifiers select bytes in blocks of 16, 32, or 64 bytes.
// Each scan handles its initial block, aligned interior blocks, and final
// overlapping block. SSE4.2 dispatches to SSSE3; VBMI dispatches to AVX-512.
//
// Scans return Some(index) for a selected byte, or None when none is found.
// Empty slices return None. Reverse traversal never constructs
// a pointer before the slice. REVERT selects nonmembers without changing
// scan direction. Mask functions are called directly and inlined.
//
// Callers must establish the instruction set required by each classifier.
// Mask bit i corresponds to loaded byte i. The classifier selects members
// when REVERT is false and nonmembers when it is true, so scans use the mask
// directly. Forward selects the lowest set bit; reverse the highest.
// Intrinsics prefixed _mm / _mm256 / _mm512 operate on 16 / 32 / 64 bytes.
// Byte shuffle instructions still use separate 16-byte tables in
// each 128-bit lane, even when the overall register is wider.
