// This private reference crate exposes upstream's Linux x86-64 fat runtime.
// Other targets benchmark the Rust algorithms without a native comparison.
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use std::{ffi::c_void, ptr::NonNull};

/// An instruction set built by Hyperscan's upstream fat runtime.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum Backend {
    Ssse3,
    Sse42,
    Avx2,
    Avx512,
    Avx512Vbmi,
}

impl Backend {
    pub fn available() -> impl Iterator<Item = Self> {
        [
            Self::Ssse3,
            Self::Sse42,
            Self::Avx2,
            Self::Avx512,
            Self::Avx512Vbmi,
        ]
        .into_iter()
        .filter(|backend| backend.is_available())
    }

    pub fn is_available(self) -> bool {
        match self {
            Self::Ssse3 => is_x86_feature_detected!("ssse3"),
            Self::Sse42 => {
                Self::Ssse3.is_available()
                    && is_x86_feature_detected!("sse4.2")
                    && is_x86_feature_detected!("popcnt")
            }
            // Upstream uses -march=core-avx2 and later CPU targets. Its kernels
            // may therefore contain BMI shifts and bit scans in addition to AVX2.
            Self::Avx2 => {
                Self::Sse42.is_available()
                    && is_x86_feature_detected!("avx2")
                    && is_x86_feature_detected!("bmi1")
                    && is_x86_feature_detected!("bmi2")
                    && is_x86_feature_detected!("lzcnt")
            }
            Self::Avx512 => {
                Self::Avx2.is_available()
                    && is_x86_feature_detected!("avx512f")
                    && is_x86_feature_detected!("avx512bw")
                    && is_x86_feature_detected!("avx512vl")
                    && is_x86_feature_detected!("avx512dq")
            }
            Self::Avx512Vbmi => {
                Self::Avx512.is_available() && is_x86_feature_detected!("avx512vbmi")
            }
        }
    }
}

#[derive(Clone, Copy)]
#[repr(u32)]
pub enum Algorithm {
    Shufti,
    Truffle,
}

/// A prepared byte set using the original Hyperscan search functions.
pub struct Matcher(NonNull<c_void>);

impl Matcher {
    /// Returns None when the backend is unavailable or the set cannot be represented.
    pub fn new(backend: Backend, algorithm: Algorithm, needle: &[u8]) -> Option<Self> {
        if !backend.is_available() {
            return None;
        }
        // SAFETY: the slice is valid for len bytes and both enum values match
        // the bridge. The returned allocation belongs exclusively to this object.
        NonNull::new(unsafe {
            scan_new(
                algorithm as u32,
                backend as u32,
                needle.as_ptr(),
                needle.len(),
            )
        })
        .map(Self)
    }

    pub fn find(&self, haystack: &[u8]) -> Option<usize> {
        // SAFETY: construction validated the backend; the bridge handles empty
        // input and forward kernels only read the supplied range.
        let index = unsafe { scan_find(self.0.as_ptr(), haystack.as_ptr(), haystack.len()) };
        (index != haystack.len()).then_some(index)
    }

    /// Returns the last matching byte's index.
    ///
    /// # Safety
    ///
    /// For nonempty input, at least one byte immediately before haystack must
    /// belong to the same allocation. Upstream forms haystack.as_ptr() - 1 as
    /// its exhausted-search sentinel. The prefix is not read.
    pub unsafe fn rfind(&self, haystack: &[u8]) -> Option<usize> {
        // SAFETY: construction validated the backend and the caller provides
        // prefix storage required for the original reverse kernel's sentinel.
        let index = unsafe { scan_rfind(self.0.as_ptr(), haystack.as_ptr(), haystack.len()) };
        (index != haystack.len()).then_some(index)
    }
}

impl Drop for Matcher {
    fn drop(&mut self) {
        // SAFETY: this is the unique pointer returned by scan_new and no calls
        // can borrow the matcher while it is being dropped.
        unsafe { scan_free(self.0.as_ptr()) }
    }
}

extern "C" {
    fn scan_new(algorithm: u32, backend: u32, needle: *const u8, len: usize) -> *mut c_void;
    fn scan_find(matcher: *const c_void, haystack: *const u8, len: usize) -> usize;
    fn scan_rfind(matcher: *const c_void, haystack: *const u8, len: usize) -> usize;
    fn scan_free(matcher: *mut c_void);
}

#[cfg(test)]
mod tests {
    use super::{Algorithm, Backend, Matcher};

    #[test]
    fn new_rejects_an_unavailable_backend() {
        for backend in [
            Backend::Ssse3,
            Backend::Sse42,
            Backend::Avx2,
            Backend::Avx512,
            Backend::Avx512Vbmi,
        ] {
            if !backend.is_available() {
                assert!(Matcher::new(backend, Algorithm::Truffle, b"a").is_none());
            }
        }
    }

    #[test]
    fn find_uses_the_prepared_needle() {
        for backend in Backend::available() {
            for algorithm in [Algorithm::Shufti, Algorithm::Truffle] {
                let matcher = Matcher::new(backend, algorithm, b"az").unwrap();
                assert_eq!(matcher.find(b"!z!a!"), Some(1));
            }
        }
    }

    #[test]
    fn find_accepts_an_empty_slice() {
        for backend in Backend::available() {
            for algorithm in [Algorithm::Shufti, Algorithm::Truffle] {
                let matcher = Matcher::new(backend, algorithm, b"a").unwrap();
                assert_eq!(matcher.find(b""), None);
            }
        }
    }

    #[test]
    fn rfind_returns_the_last_member() {
        let storage = *b" a!z!";
        for backend in Backend::available() {
            for algorithm in [Algorithm::Shufti, Algorithm::Truffle] {
                let matcher = Matcher::new(backend, algorithm, b"az").unwrap();
                // SAFETY: storage[0] supplies the required prefix byte.
                assert_eq!(unsafe { matcher.rfind(&storage[1..]) }, Some(2));
            }
        }
    }

    #[test]
    fn rfind_does_not_match_the_prefix_byte() {
        let storage = *b"a!!!";
        for backend in Backend::available() {
            for algorithm in [Algorithm::Shufti, Algorithm::Truffle] {
                let matcher = Matcher::new(backend, algorithm, b"a").unwrap();
                // SAFETY: storage[0] belongs to the same allocation but is not
                // part of the searched slice. It is the exhausted sentinel.
                assert_eq!(unsafe { matcher.rfind(&storage[1..]) }, None);
            }
        }
    }
}
