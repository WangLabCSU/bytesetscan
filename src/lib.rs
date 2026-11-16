#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]

mod backend;
pub mod bitset;
pub mod shufti;
pub mod truffle;

pub use self::backend::SimdBackend;

#[cfg(test)]
mod tests {
    extern crate std;

    use std::{vec, vec::Vec};

    // Needle/haystack pairs from Hyperscan's Shufti and Truffle search tests.
    // Keep project-specific cases in the tests for their own methods.
    // Source revision: bc3b191ab56055e8560c7cdc161c289c4d76e3d2.
    // https://github.com/intel/hyperscan/blob/bc3b191ab56055e8560c7cdc161c289c4d76e3d2/unit/internal/shufti.cpp
    // https://github.com/intel/hyperscan/blob/bc3b191ab56055e8560c7cdc161c289c4d76e3d2/unit/internal/truffle.cpp
    pub(crate) fn hyperscan_needle_and_haystack() -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut inputs = Vec::new();

        // Forward tests advance the beginning of the searched slice.
        let forward16: &[(&[u8], &[u8])] = &[
            (
                b"aB",
                b"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ),
            (
                b"V",
                b"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            ),
            (
                b"a",
                b"bbbbbbbbbbbbbbbbbaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbabbbbbbbbbbbb",
            ),
            (
                b"aB",
                b"bbbbbbbbbbbbbbbbbBaaaaaaaaaaaaaaabbbbbbbbbbbbbbbabbbbbbbbbbbb",
            ),
            (
                b"aCAc",
                b"bbbbbbbbbbbbbbbbbAaaaaaaaaaaaaaaabbbbbbbbbbbbbbbabbbbbbbbbbbb",
            ),
            (
                b"aCAc",
                b"bbbbbbbbbbbbbbbbbCaaaaaaaaaaaaaaabbbbbbbbbbbbbbbabbbbbbbbbbbb",
            ),
            (
                b"aCAc",
                b"bbbbbbbbbbbbbbbbbcaaaaaaaaaaaaaaabbbbbbbbbbbbbbbabbbbbbbbbbbb",
            ),
            (
                b"aCAc",
                b"bbbbbbbbbbbbbbbbbaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbabbbbbbbbbbbb",
            ),
            (
                b"a",
                b"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\xff",
            ),
            (
                b"a",
                b"bbbbbbbbbbbbbbbbbabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbabbbbbbbbbbbb",
            ),
        ];
        for &(needle, haystack) in forward16 {
            for start in 0..16 {
                inputs.push((needle.to_vec(), haystack[start..].to_vec()));
            }
        }

        let forward32: &[(&[u8], &[u8])] = &[
            (
                b"a",
                b"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ),
            (
                b"a",
                b"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbabbbbbbbbbbbbbbabbbbbbbbbbbb",
            ),
        ];
        for &(needle, haystack) in forward32 {
            for start in 0..32 {
                inputs.push((needle.to_vec(), haystack[start..].to_vec()));
            }
        }

        // Reverse tests shorten the slice's end. These strings already omit
        // the leading byte excluded by the original no-match calls.
        let reverse16: &[(&[u8], &[u8])] = &[
            (
                b"a",
                b"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ),
            (
                b"aB",
                b"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ),
            (
                b"V",
                b"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            ),
            (
                b"a",
                b"bbbbbbabbbbbbbbbbabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ),
            (
                b"a",
                b"bbbbabbbbbbbbbbbbaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ),
            (
                b"aB",
                b"bbbbbbbbbbbbbbbbbaaaaaaaaaaaaaaaBbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ),
            (
                b"aB",
                b"bbbbbbbbbbbbbbbbbaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ),
            (
                b"aCAc",
                b"bbbbbbbbbbbbbbbbbaaaaaaaaaaaaaaaAbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ),
            (
                b"aCAc",
                b"bbbbbbbbbbbbbbbbbaaaaaaaaaaaaaaaCbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ),
            (
                b"aCAc",
                b"bbbbbbbbbbbbbbbbbaaaaaaaaaaaaaaacbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ),
            (
                b"aCAc",
                b"bbbbbbbbbbbbbbbbbaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ),
        ];
        for &(needle, haystack) in reverse16 {
            for trim in 0..16 {
                inputs.push((needle.to_vec(), haystack[..haystack.len() - trim].to_vec()));
            }
        }

        // Short inputs include embedded NUL bytes; their lengths are explicit.
        let short: &[(&[u8], &[u8])] = &[
            (b"a", b"a"),
            (b"a", b"bbbbbbbabbb"),
            (b"\0", b"bbbbbbb\0bbb"),
            (b"a", b"\0\0\0\0\0\0\0a\0\0\0"),
            (b"a", b"bbbbbbbabbbb"),
            (b"a", b"babbbbbabbbb"),
        ];
        for &(needle, haystack) in short {
            inputs.push((needle.to_vec(), haystack.to_vec()));
        }

        // The original 400-byte buffer is searched only through byte 398.
        let mut haystack = vec![b'b'; 399];
        haystack[120] = b'a';
        for start in 0..16 {
            inputs.push((b"a".to_vec(), haystack[start..].to_vec()));
        }

        // Preserve accumulated matches as each source loop modifies its buffer.
        let mut haystack = vec![b'b'; 76];
        for step in 0..31 {
            haystack[48 - step] = b'a';
            inputs.push((b"a".to_vec(), haystack.clone()));
        }
        for len in [76, 256] {
            let mut haystack = vec![b'b'; len];
            for position in 0..len {
                haystack[position] = b'a';
                inputs.push((b"a".to_vec(), haystack.clone()));
            }
        }

        // Truffle exercises every member of these two inclusive byte ranges.
        let needle: Vec<u8> = (b'0'..=b'Z').collect();
        let mut haystack = vec![b'*'; 128];
        for &byte in &needle {
            haystack[17] = byte;
            inputs.push((needle.clone(), haystack.clone()));
        }
        let needle: Vec<u8> = (127..=255).collect();
        let mut haystack = vec![b'*'; 128];
        for &byte in &needle {
            haystack[40] = byte;
            inputs.push((needle.clone(), haystack.clone()));
        }

        // Forward/reverse and Shufti/Truffle tests share many identical pairs.
        inputs.sort_unstable();
        inputs.dedup();
        inputs
    }

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
