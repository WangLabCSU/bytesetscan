#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unnameable_types, unreachable_pub)]

mod backend;
pub mod shufti;

pub use self::backend::SimdBackend;

#[cfg(test)]
mod tests {
    extern crate std;

    use crate::backend::SimdBackend;

    // One writable page between two inaccessible pages.
    #[cfg(target_os = "linux")]
    pub(crate) struct GuardedPage {
        mapping: *mut std::ffi::c_void,
        page_size: usize,
    }

    #[cfg(target_os = "linux")]
    impl GuardedPage {
        pub(crate) fn new() -> Self {
            use std::{ffi::c_void, ptr};

            extern "C" {
                fn getpagesize() -> i32;
                fn mmap(
                    addr: *mut c_void,
                    len: usize,
                    prot: i32,
                    flags: i32,
                    fd: i32,
                    offset: isize,
                ) -> *mut c_void;
                fn mprotect(addr: *mut c_void, len: usize, prot: i32) -> i32;
            }

            // SAFETY: getpagesize has no arguments or preconditions.
            let page_size = unsafe { getpagesize() };
            assert!(page_size > 0);
            let page_size = page_size as usize;
            // mmap's second argument is a byte count: page_size * 3 maps three
            // contiguous pages. PROT_NONE initially makes all three inaccessible.
            // Linux: PROT_NONE = 0, MAP_PRIVATE = 2, MAP_ANONYMOUS = 0x20.
            // SAFETY: request a new anonymous mapping; no existing memory is replaced.
            let mapping = unsafe { mmap(ptr::null_mut(), page_size * 3, 0, 2 | 0x20, -1, 0) };
            assert_ne!(mapping as isize, -1, "mmap failed");
            // Own the mapping before mprotect so a failure also releases it.
            let guarded = Self { mapping, page_size };
            // mapping points to the first page. Casting to u8 makes add count
            // bytes: adding page_size skips one page and reaches the start of
            // the second (middle) page.
            // SAFETY: this one-page offset stays inside the three-page mapping.
            let middle = unsafe { mapping.cast::<u8>().add(page_size) };
            // Grant read/write access (PROT_READ | PROT_WRITE = 3) only to the
            // middle page, leaving this layout: [guard][read/write][guard].
            // SAFETY: middle is page-aligned, and the entire page lies inside
            // the mapping we own.
            assert_eq!(unsafe { mprotect(middle.cast(), page_size, 3) }, 0);
            guarded
        }

        pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
            // SAFETY: new() made the middle page writable, and anonymous mmap
            // initialized its bytes to zero. Borrowing self exclusively prevents
            // aliasing and keeps the mapping alive for the returned slice.
            unsafe {
                std::slice::from_raw_parts_mut(
                    self.mapping.cast::<u8>().add(self.page_size),
                    self.page_size,
                )
            }
        }
    }

    #[cfg(target_os = "linux")]
    impl Drop for GuardedPage {
        fn drop(&mut self) {
            extern "C" {
                fn munmap(addr: *mut std::ffi::c_void, len: usize) -> i32;
            }
            // SAFETY: self owns the original mapping address and all three pages.
            assert_eq!(unsafe { munmap(self.mapping, self.page_size * 3) }, 0);
        }
    }

    pub(crate) fn supported_backends() -> impl Iterator<Item = SimdBackend> {
        backends().filter(|backend| backend.is_available())
    }

    pub(crate) fn backends() -> impl Iterator<Item = SimdBackend> {
        [
            SimdBackend::Fallback,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SimdBackend::Ssse3,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SimdBackend::Sse4_2,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SimdBackend::Avx2,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SimdBackend::Avx512,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SimdBackend::Avx512Vbmi,
            #[cfg(target_arch = "aarch64")]
            SimdBackend::Neon,
            #[cfg(target_arch = "wasm32")]
            SimdBackend::WasmSimd128,
        ]
        .into_iter()
    }
}
