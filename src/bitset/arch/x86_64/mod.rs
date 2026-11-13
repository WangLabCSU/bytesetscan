mod avx2;
mod avx512;
mod sse42;
mod ssse3;
mod vbmi;

pub(super) use self::{
    avx2::{bitset_avx2, rbitset_avx2},
    avx512::{bitset_avx512, rbitset_avx512},
    sse42::{bitset_sse42, rbitset_sse42},
    ssse3::{bitset_ssse3, rbitset_ssse3},
    vbmi::{bitset_vbmi, rbitset_vbmi},
};

// x86 stores each u64 least-significant byte first, so the original
// bitmap bytes are already in the order required by the SIMD lookups.
