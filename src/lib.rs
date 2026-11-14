#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unnameable_types, unreachable_pub)]

mod backend;
pub mod bitset;
pub mod shufti;
pub mod truffle;

pub use self::backend::SimdBackend;

#[cfg(test)]
mod tests {
    extern crate std;

    // One writable page between two inaccessible pages.
    #[cfg(target_os = "linux")]
    pub(crate) struct GuardedPage {
        mapping: *mut std::ffi::c_void,
        page_size: usize,
    }

    #[cfg(target_os = "linux")]
    impl GuardedPage {
        pub(crate) fn new() -> Self {
            use std::ptr;

            // SAFETY: _SC_PAGESIZE is a valid sysconf query and takes no pointers.
            let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            assert!(page_size > 0);
            let page_size = page_size as usize;
            // mmap's second argument is a byte count: page_size * 3 maps three
            // contiguous pages. PROT_NONE initially makes all three inaccessible.
            // libc supplies the target ABI's flags; MAP_ANONYMOUS differs
            // between Linux architectures, so its numeric value is not portable.
            // SAFETY: request a new anonymous mapping; no existing memory is replaced.
            let mapping = unsafe {
                libc::mmap(
                    ptr::null_mut(),
                    page_size * 3,
                    libc::PROT_NONE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            assert_ne!(mapping, libc::MAP_FAILED, "mmap failed");
            // Own the mapping before mprotect so a failure also releases it.
            let guarded = Self { mapping, page_size };
            // mapping points to the first page. Casting to u8 makes add count
            // bytes: adding page_size skips one page and reaches the start of
            // the second (middle) page.
            // SAFETY: this one-page offset stays inside the three-page mapping.
            let middle = unsafe { mapping.cast::<u8>().add(page_size) };
            // Grant read/write access only to the middle page, leaving
            // this layout: [guard][read/write][guard].
            // SAFETY: middle is page-aligned, and the entire page lies inside
            // the mapping we own.
            assert_eq!(
                unsafe {
                    libc::mprotect(middle.cast(), page_size, libc::PROT_READ | libc::PROT_WRITE)
                },
                0
            );
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
            // SAFETY: self owns the original mapping address and all three pages.
            assert_eq!(unsafe { libc::munmap(self.mapping, self.page_size * 3) }, 0);
        }
    }
}
