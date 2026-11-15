// Automatic matchers use backend selection to initialize search dispatch.

/// Identifies a search backend.
///
/// SIMD variants are specific to the target architecture. A variant can be
/// constructed even when the current processor cannot use that backend.
///
/// Uses a `u8` representation, with the size and alignment of `u8`.
// Discriminants start at zero for Fallback and follow declaration order after
// cfg filtering; new() defines selection priority separately.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SimdBackend {
    /// The portable scalar implementation.
    Fallback,
    /// The SSSE3 backend for x86 and x86-64.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Ssse3,
    /// The SSE4.2 backend, which also requires SSSE3 and POPCNT.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Sse4_2,
    /// The AVX2 backend for x86 and x86-64.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2,
    /// The AVX-512F/BW backend, which also requires AVX2.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx512,
    /// The AVX-512 VBMI backend, which also requires AVX2 and AVX-512F/BW.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx512Vbmi,
    /// The NEON backend for AArch64.
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// The SIMD128 backend for WebAssembly.
    #[cfg(target_arch = "wasm32")]
    WasmSimd128,
}

impl SimdBackend {
    /// Creates the highest-priority backend supported by the current environment.
    ///
    /// Returns [`Self::Fallback`] when no SIMD backend is available.
    ///
    /// See [`Self::is_available`] for how backend availability is determined.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytesetscan::SimdBackend;
    ///
    /// let backend = SimdBackend::new();
    /// assert!(backend.is_available());
    /// ```
    #[inline]
    pub fn new() -> Self {
        // Array order defines preference, independently of enum discriminants.
        // is_available() supplies the platform's runtime or target-feature
        // checks. cfg removes backends for other architectures; the explicit
        // element type also permits an empty array on targets with no SIMD backend.
        let backends: [Self; _] = [
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx512Vbmi,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx512,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx2,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Sse4_2,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Ssse3,
            #[cfg(target_arch = "aarch64")]
            Self::Neon,
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128,
        ];
        for backend in backends {
            if backend.is_available() {
                return backend;
            }
        }
        Self::Fallback
    }

    /// Returns an iterator over all backends defined for the compilation target.
    ///
    /// Includes [`Self::Fallback`] and SIMD backends that may be unavailable in
    /// the current environment. Use [`Self::is_available`] to filter them.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytesetscan::SimdBackend;
    ///
    /// let available: Vec<_> = SimdBackend::backends()
    ///     .filter(SimdBackend::is_available)
    ///     .collect();
    /// assert!(available.contains(&SimdBackend::Fallback));
    /// ```
    pub fn backends() -> impl Iterator<Item = Self> {
        [
            Self::Fallback,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Ssse3,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Sse4_2,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx2,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx512,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx512Vbmi,
            #[cfg(target_arch = "aarch64")]
            Self::Neon,
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128,
        ]
        .into_iter()
    }

    /// Returns whether this backend can run in the current environment.
    ///
    /// With `std`, checks processor and operating-system support on x86,
    /// x86-64, and AArch64. Without `std`, returns whether the compilation
    /// target enables all features required by this backend.
    ///
    /// On WebAssembly, uses the compilation target's `simd128` feature in
    /// either build. [`Self::Fallback`] always returns `true`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytesetscan::SimdBackend;
    ///
    /// assert!(SimdBackend::Fallback.is_available());
    /// ```
    #[inline]
    pub fn is_available(&self) -> bool {
        #[cfg(not(feature = "std"))]
        {
            // Does not perform runtime detection. A `false` result does not imply that
            // the current processor lacks support.
            match self {
                Self::Fallback => true,
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                Self::Ssse3 => cfg!(target_feature = "ssse3"),
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                Self::Sse4_2 => cfg!(all(
                    target_feature = "ssse3",
                    target_feature = "sse4.2",
                    target_feature = "popcnt"
                )),
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                Self::Avx2 => cfg!(target_feature = "avx2"),
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                Self::Avx512 => cfg!(all(
                    target_feature = "avx2",
                    target_feature = "avx512f",
                    target_feature = "avx512bw"
                )),
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                Self::Avx512Vbmi => cfg!(all(
                    target_feature = "avx2",
                    target_feature = "avx512f",
                    target_feature = "avx512bw",
                    target_feature = "avx512vbmi"
                )),
                #[cfg(target_arch = "aarch64")]
                Self::Neon => cfg!(target_feature = "neon"),
                #[cfg(target_arch = "wasm32")]
                Self::WasmSimd128 => cfg!(target_feature = "simd128"),
            }
        }

        #[cfg(feature = "std")]
        {
            // std validates CPU and OS support, compiler feature implications,
            // and platform restrictions.
            // https://doc.rust-lang.org/src/std_detect/detect/os/x86.rs.html
            // std caches processor capabilities used by these checks.
            match self {
                Self::Fallback => true,
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                Self::Ssse3 => std::is_x86_feature_detected!("ssse3"),
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                Self::Sse4_2 => {
                    std::is_x86_feature_detected!("ssse3")
                        && std::is_x86_feature_detected!("sse4.2")
                        && std::is_x86_feature_detected!("popcnt")
                }
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                Self::Avx2 => std::is_x86_feature_detected!("avx2"),
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                Self::Avx512 => {
                    std::is_x86_feature_detected!("avx2")
                        && std::is_x86_feature_detected!("avx512f")
                        && std::is_x86_feature_detected!("avx512bw")
                }
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                Self::Avx512Vbmi => {
                    Self::Avx512.is_available() && std::is_x86_feature_detected!("avx512vbmi")
                }
                #[cfg(target_arch = "aarch64")]
                Self::Neon => std::arch::is_aarch64_feature_detected!("neon"),
                #[cfg(target_arch = "wasm32")]
                Self::WasmSimd128 => {
                    // WebAssembly has no in-module runtime SIMD detection.
                    // The SIMD kernels are compiled only when simd128 is enabled.
                    cfg!(target_feature = "simd128")
                }
            }
        }
    }
}

impl Default for SimdBackend {
    /// Returns the same backend as [`Self::new`].
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::SimdBackend;

    // Both builds check each backend before exercising its SIMD kernels.
    // no_std tests include every backend enabled by the compilation target.
    #[test]
    fn new_selects_the_highest_priority_available_backend() {
        let selected = SimdBackend::new();
        let expected = SimdBackend::backends()
            .filter(SimdBackend::is_available)
            .last()
            .unwrap();
        assert_eq!(selected, expected);
    }
}
