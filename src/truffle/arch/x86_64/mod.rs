mod avx2;
mod avx512;
mod sse42;
mod ssse3;
mod vbmi;

pub(super) use self::{
    avx2::{rtruffle_avx2, truffle_avx2},
    avx512::{rtruffle_avx512, truffle_avx512},
    sse42::{rtruffle_sse42, truffle_sse42},
    ssse3::{rtruffle_ssse3, truffle_ssse3},
    vbmi::{rtruffle_vbmi, truffle_vbmi},
};

// The mask classifiers identify nonmembers in blocks of 16, 32, or 64 bytes.
// Each scan handles its initial block, aligned interior blocks, and final
// overlapping block. SSE4.2 shares the SSSE3 scan; VBMI shares the AVX-512 scan.
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
