mod iter;
mod vtable;

pub use self::iter::{TruffleEngineFinder, TruffleEngineMismatcher};

use self::vtable::VTable;
use super::TruffleTable;
use crate::SimdBackend;

/// A Truffle search engine with a custom backend.
///
/// # Examples
///
/// ```
/// use bytesetscan::{truffle::TruffleEngine, SimdBackend};
///
/// let engine = TruffleEngine::new(SimdBackend::Fallback, b"a").unwrap();
/// assert_eq!(engine.find(b"a!a"), Some(0));
/// assert_eq!(engine.mismatch(b"a!a"), Some(1));
/// assert_eq!(engine.rfind(b"a!a"), Some(2));
/// assert_eq!(engine.rmismatch(b"a!a"), Some(1));
/// ```
#[derive(Debug, Clone)]
pub struct TruffleEngine {
    table: TruffleTable,
    vtable: VTable,
}

impl TruffleEngine {
    /// Creates an engine for `needle` using `backend`.
    ///
    /// Returns `None` if the backend is unavailable.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytesetscan::{truffle::TruffleEngine, SimdBackend};
    ///
    /// let engine = TruffleEngine::new(SimdBackend::Fallback, b" \t").unwrap();
    /// assert_eq!(engine.find(b"hello world"), Some(5));
    /// ```
    pub fn new(backend: SimdBackend, needle: &[u8]) -> Option<Self> {
        Self::with_table(TruffleTable::new(needle), backend)
    }

    /// Creates an engine that can ignore ASCII letter case.
    ///
    /// When `ignore_case` is `true`, ASCII letters match either case; other
    /// bytes match exactly. When it is `false`, this behaves like [`Self::new`].
    ///
    /// Returns `None` if the backend is unavailable.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytesetscan::{truffle::TruffleEngine, SimdBackend};
    ///
    /// let letters = TruffleEngine::with_case(true, SimdBackend::Fallback, b"a").unwrap();
    /// assert_eq!(letters.find(b"!A!"), Some(1));
    /// ```
    pub fn with_case(ignore_case: bool, backend: SimdBackend, needle: &[u8]) -> Option<Self> {
        Self::with_table(TruffleTable::with_case(ignore_case, needle), backend)
    }

    // Move the prepared masks into the engine without rebuilding them. VTable
    // validates the backend before any callable engine can be returned.
    fn with_table(table: TruffleTable, backend: SimdBackend) -> Option<Self> {
        Some(Self {
            table,
            vtable: VTable::new(backend)?,
        })
    }

    /// Returns the index of the first byte in `haystack` that belongs to the needle.
    ///
    /// Returns `None` if no match is found.
    #[inline]
    pub fn find(&self, haystack: &[u8]) -> Option<usize> {
        self.vtable.search::<false>(&self.table, haystack)
    }

    /// Returns the index of the first byte in `haystack` that is not in the needle.
    ///
    /// Returns `None` if all bytes belong to the needle.
    #[inline]
    pub fn mismatch(&self, haystack: &[u8]) -> Option<usize> {
        self.vtable.search::<true>(&self.table, haystack)
    }

    /// Returns the index of the last byte in `haystack` that belongs to the needle.
    ///
    /// Searches from the end of the slice. Returns `None` if no match is found.
    #[inline]
    pub fn rfind(&self, haystack: &[u8]) -> Option<usize> {
        self.vtable.rsearch::<false>(&self.table, haystack)
    }

    /// Returns the index of the last byte in `haystack` that is not in the needle.
    ///
    /// Searches from the end of the slice. Returns `None` if all bytes belong
    /// to the needle.
    #[inline]
    pub fn rmismatch(&self, haystack: &[u8]) -> Option<usize> {
        self.vtable.rsearch::<true>(&self.table, haystack)
    }

    /// Returns an iterator over the indices of bytes belonging to the needle.
    ///
    /// Indices are yielded in ascending order. Use [`.rev()`](Iterator::rev)
    /// to iterate in reverse.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytesetscan::{truffle::TruffleEngine, SimdBackend};
    ///
    /// let engine = TruffleEngine::new(SimdBackend::Fallback, b"a").unwrap();
    /// let mut matches = engine.finder(b"banana");
    /// assert_eq!(matches.next(), Some(1));
    /// assert_eq!(matches.next_back(), Some(5));
    /// assert!(matches.eq([3]));
    /// ```
    #[inline]
    pub fn finder<'a, 'h>(&'a self, haystack: &'h [u8]) -> TruffleEngineFinder<'a, 'h> {
        TruffleEngineFinder::new(self, haystack)
    }

    /// Returns an iterator over the indices of bytes not belonging to the needle.
    ///
    /// Indices are yielded in ascending order. Use [`.rev()`](Iterator::rev)
    /// to iterate in reverse.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytesetscan::{truffle::TruffleEngine, SimdBackend};
    ///
    /// let engine = TruffleEngine::with_case(true, SimdBackend::Fallback, b"ACGT").unwrap();
    /// assert!(engine.mismatcher(b"ac?GT!").eq([2, 5]));
    /// assert!(engine.mismatcher(b"ac?GT!").rev().eq([5, 2]));
    /// ```
    #[inline]
    pub fn mismatcher<'a, 'h>(&'a self, haystack: &'h [u8]) -> TruffleEngineMismatcher<'a, 'h> {
        TruffleEngineMismatcher::new(self, haystack)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::{vec, vec::Vec};

    use super::TruffleEngine;
    use crate::tests::hyperscan_needle_and_haystack;
    use crate::SimdBackend;

    #[test]
    fn find_returns_the_first_member() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            let matcher = TruffleEngine::new(backend, b"Ab").unwrap();
            // Both positions qualify; the earlier one must win.
            let mut haystack = [b'!'; 128];
            haystack[17] = b'A';
            haystack[96] = b'b';

            assert_eq!(matcher.find(&haystack), Some(17));
        }
    }

    #[test]
    fn find_recognizes_every_byte_value() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            let haystack: [u8; 256] = core::array::from_fn(|index| index as u8);
            for member in 0..=255u8 {
                let matcher = TruffleEngine::new(backend, &[member]).unwrap();
                // Each byte occurs once, at the index equal to its value.
                assert_eq!(
                    matcher.find(&haystack),
                    Some(usize::from(member)),
                    "member={member}"
                );
            }
        }
    }

    #[test]
    fn find_distinguishes_members_of_a_sparse_set() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            // h << 4 fills the high nibble; adding h fills the low nibble.
            // These isolated members span the whole byte range.
            let needle: Vec<u8> = (0..16).map(|h| (h << 4) + h).collect();
            let matcher = TruffleEngine::new(backend, &needle).unwrap();
            for byte in 0..=255 {
                let expected = (needle.contains(&byte)).then_some(0);
                // A uniform input either qualifies everywhere or nowhere.
                assert_eq!(matcher.find(&[byte; 128]), expected, "byte={byte}");
            }
        }
    }

    #[test]
    fn find_distinguishes_nonmembers_of_a_dense_set() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            // Exclude bytes whose high and low nibbles are equal.
            let needle: Vec<u8> = (0..=255u8)
                .filter(|byte| byte >> 4 != byte & 0b00001111)
                .collect();
            let matcher = TruffleEngine::new(backend, &needle).unwrap();
            for byte in 0..=255 {
                let expected = (needle.contains(&byte)).then_some(0);
                // A uniform input either qualifies everywhere or nowhere.
                assert_eq!(matcher.find(&[byte; 128]), expected, "byte={byte}");
            }
        }
    }

    #[test]
    fn find_uses_ascii_case_expansion() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            let matcher = TruffleEngine::with_case(true, backend, b"A").unwrap();
            assert_eq!(matcher.find(b"!a!A!"), Some(1));
        }
    }

    #[test]
    fn find_returns_none_for_an_empty_haystack() {
        let all_bytes: Vec<u8> = (0..=255).collect();
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            for needle in [b"".as_slice(), b"a", all_bytes.as_slice()] {
                let matcher = TruffleEngine::new(backend, needle).unwrap();
                assert_eq!(matcher.find(b""), None, "needle={needle:?}");
            }
        }
    }

    #[test]
    fn find_returns_none_for_an_empty_needle() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            let matcher = TruffleEngine::new(backend, b"").unwrap();
            assert_eq!(matcher.find(b"abc"), None);
            // Also search all byte values, spanning multiple SIMD blocks.
            let all_bytes: Vec<u8> = (0..=255).collect();
            assert_eq!(matcher.find(&all_bytes), None);
        }
    }

    #[test]
    fn find_returns_first_position_for_a_full_needle() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            let needle: Vec<u8> = (0..=255).collect();
            let matcher = TruffleEngine::new(backend, &needle).unwrap();
            assert_eq!(matcher.find(b"abc"), Some(0));
            // Also search all byte values, spanning multiple SIMD blocks.
            assert_eq!(matcher.find(&needle), Some(0));
        }
    }

    #[test]
    fn find_ignores_matches_outside_the_slice() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            let matcher = TruffleEngine::new(backend, &[0]).unwrap();
            // Cover short inputs, vector tails, and every 64-byte alignment.
            for len in 0..=257 {
                for offset in 0..64 {
                    // Only the surrounding zeros match the needle.
                    let mut storage = vec![0; offset + len + 64];
                    let haystack = &mut storage[offset..offset + len];
                    haystack.fill(1);
                    assert_eq!(matcher.find(haystack), None, "len={len}, offset={offset}");
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn find_stays_inside_guarded_slices() {
        use crate::tests::GuardedPage;

        let mut guarded = GuardedPage::new();
        let page = guarded.as_mut_slice();
        // No members: the search must read through the whole slice.
        page.fill(1);
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            let matcher = TruffleEngine::new(backend, &[0]).unwrap();
            for len in 0..=257 {
                // Start immediately after the leading guard page.
                assert_eq!(matcher.find(&page[..len]), None, "start, len={len}");
                // End immediately before the trailing guard page.
                assert_eq!(
                    matcher.find(&page[page.len() - len..]),
                    None,
                    "end, len={len}"
                );
            }
        }
    }

    #[test]
    fn find_returns_expected_positions() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            for (needle, haystack) in hyperscan_needle_and_haystack() {
                let matcher = TruffleEngine::new(backend, &needle).unwrap();
                let expected = haystack.iter().position(|byte| needle.contains(byte));
                // Exercise every start alignment of the widest (64-byte) vector.
                let mut storage = vec![0; haystack.len() + 63];
                for offset in 0..64 {
                    let input = &mut storage[offset..offset + haystack.len()];
                    input.copy_from_slice(&haystack);
                    assert_eq!(
                        matcher.find(input),
                        expected,
                        "backend={backend:?}, needle={needle:?}, haystack={haystack:?}, offset={offset}"
                    );
                }
            }
        }
    }

    #[test]
    fn mismatch_returns_the_first_nonmember() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            let matcher = TruffleEngine::new(backend, b"Ab").unwrap();
            // Both positions qualify; the earlier one must win.
            let mut haystack = [b'A'; 128];
            haystack[17] = b'!';
            haystack[96] = b'?';

            assert_eq!(matcher.mismatch(&haystack), Some(17));
        }
    }

    #[test]
    fn mismatch_recognizes_every_byte_value() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            for member in 0..=255u8 {
                let matcher = TruffleEngine::new(backend, &[member]).unwrap();
                let mut haystack = [member; 129];
                // The only nonmember lies just beyond a 64-byte block.
                haystack[65] = member.wrapping_add(1);
                assert_eq!(matcher.mismatch(&haystack), Some(65), "member={member}");
            }
        }
    }

    #[test]
    fn mismatch_distinguishes_members_of_a_sparse_set() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            // h << 4 fills the high nibble; adding h fills the low nibble.
            // These isolated members span the whole byte range.
            let needle: Vec<u8> = (0..16).map(|h| (h << 4) + h).collect();
            let matcher = TruffleEngine::new(backend, &needle).unwrap();
            for byte in 0..=255 {
                let expected = (!needle.contains(&byte)).then_some(0);
                // A uniform input either qualifies everywhere or nowhere.
                assert_eq!(matcher.mismatch(&[byte; 128]), expected, "byte={byte}");
            }
        }
    }

    #[test]
    fn mismatch_distinguishes_nonmembers_of_a_dense_set() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            // Exclude bytes whose high and low nibbles are equal.
            let needle: Vec<u8> = (0..=255u8)
                .filter(|byte| byte >> 4 != byte & 0b00001111)
                .collect();
            let matcher = TruffleEngine::new(backend, &needle).unwrap();
            for byte in 0..=255 {
                let expected = (!needle.contains(&byte)).then_some(0);
                // A uniform input either qualifies everywhere or nowhere.
                assert_eq!(matcher.mismatch(&[byte; 128]), expected, "byte={byte}");
            }
        }
    }

    #[test]
    fn mismatch_uses_ascii_case_expansion() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            let matcher = TruffleEngine::with_case(true, backend, b"A").unwrap();
            assert_eq!(matcher.mismatch(b"aA!aA?"), Some(2));
        }
    }

    #[test]
    fn mismatch_returns_none_for_an_empty_haystack() {
        let all_bytes: Vec<u8> = (0..=255).collect();
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            for needle in [b"".as_slice(), b"a", all_bytes.as_slice()] {
                let matcher = TruffleEngine::new(backend, needle).unwrap();
                assert_eq!(matcher.mismatch(b""), None, "needle={needle:?}");
            }
        }
    }

    #[test]
    fn mismatch_returns_first_position_for_an_empty_needle() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            let matcher = TruffleEngine::new(backend, b"").unwrap();
            assert_eq!(matcher.mismatch(b"abc"), Some(0));
            // Also search all byte values, spanning multiple SIMD blocks.
            let all_bytes: Vec<u8> = (0..=255).collect();
            assert_eq!(matcher.mismatch(&all_bytes), Some(0));
        }
    }

    #[test]
    fn mismatch_returns_none_for_a_full_needle() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            let needle: Vec<u8> = (0..=255).collect();
            let matcher = TruffleEngine::new(backend, &needle).unwrap();
            assert_eq!(matcher.mismatch(b"abc"), None);
            // Also search all byte values, spanning multiple SIMD blocks.
            assert_eq!(matcher.mismatch(&needle), None);
        }
    }

    #[test]
    fn mismatch_ignores_nonmembers_outside_the_slice() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            let matcher = TruffleEngine::new(backend, &[1]).unwrap();
            // Cover short inputs, vector tails, and every 64-byte alignment.
            for len in 0..=257 {
                for offset in 0..64 {
                    // Only the surrounding zeros are nonmembers.
                    let mut storage = vec![0; offset + len + 64];
                    let haystack = &mut storage[offset..offset + len];
                    haystack.fill(1);
                    assert_eq!(
                        matcher.mismatch(haystack),
                        None,
                        "len={len}, offset={offset}"
                    );
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn mismatch_stays_inside_guarded_slices() {
        use crate::tests::GuardedPage;

        let mut guarded = GuardedPage::new();
        let page = guarded.as_mut_slice();
        // No nonmembers: the search must read through the whole slice.
        page.fill(1);
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            let matcher = TruffleEngine::new(backend, &[1]).unwrap();
            for len in 0..=257 {
                // Start immediately after the leading guard page.
                assert_eq!(matcher.mismatch(&page[..len]), None, "start, len={len}");
                // End immediately before the trailing guard page.
                assert_eq!(
                    matcher.mismatch(&page[page.len() - len..]),
                    None,
                    "end, len={len}"
                );
            }
        }
    }

    #[test]
    fn mismatch_returns_expected_positions() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            for (needle, haystack) in hyperscan_needle_and_haystack() {
                let matcher = TruffleEngine::new(backend, &needle).unwrap();
                let expected = haystack.iter().position(|byte| !needle.contains(byte));
                // Exercise every start alignment of the widest (64-byte) vector.
                let mut storage = vec![0; haystack.len() + 63];
                for offset in 0..64 {
                    let input = &mut storage[offset..offset + haystack.len()];
                    input.copy_from_slice(&haystack);
                    assert_eq!(
                        matcher.mismatch(input),
                        expected,
                        "backend={backend:?}, needle={needle:?}, haystack={haystack:?}, offset={offset}"
                    );
                }
            }
        }
    }

    #[test]
    fn rfind_returns_the_last_member() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            let matcher = TruffleEngine::new(backend, b"Ab").unwrap();
            // Both positions qualify; the later one must win.
            let mut haystack = [b'!'; 128];
            haystack[17] = b'A';
            haystack[96] = b'b';

            assert_eq!(matcher.rfind(&haystack), Some(96));
        }
    }

    #[test]
    fn rfind_recognizes_every_byte_value() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            let haystack: [u8; 256] = core::array::from_fn(|index| index as u8);
            for member in 0..=255u8 {
                let matcher = TruffleEngine::new(backend, &[member]).unwrap();
                // Each byte occurs once, at the index equal to its value.
                assert_eq!(
                    matcher.rfind(&haystack),
                    Some(usize::from(member)),
                    "member={member}"
                );
            }
        }
    }

    #[test]
    fn rfind_distinguishes_members_of_a_sparse_set() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            // h << 4 fills the high nibble; adding h fills the low nibble.
            // These isolated members span the whole byte range.
            let needle: Vec<u8> = (0..16).map(|h| (h << 4) + h).collect();
            let matcher = TruffleEngine::new(backend, &needle).unwrap();
            for byte in 0..=255 {
                let expected = (needle.contains(&byte)).then_some(127);
                // A uniform input either qualifies everywhere or nowhere.
                assert_eq!(matcher.rfind(&[byte; 128]), expected, "byte={byte}");
            }
        }
    }

    #[test]
    fn rfind_distinguishes_nonmembers_of_a_dense_set() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            // Exclude bytes whose high and low nibbles are equal.
            let needle: Vec<u8> = (0..=255u8)
                .filter(|byte| byte >> 4 != byte & 0b00001111)
                .collect();
            let matcher = TruffleEngine::new(backend, &needle).unwrap();
            for byte in 0..=255 {
                let expected = (needle.contains(&byte)).then_some(127);
                // A uniform input either qualifies everywhere or nowhere.
                assert_eq!(matcher.rfind(&[byte; 128]), expected, "byte={byte}");
            }
        }
    }

    #[test]
    fn rfind_uses_ascii_case_expansion() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            let matcher = TruffleEngine::with_case(true, backend, b"A").unwrap();
            assert_eq!(matcher.rfind(b"!A!a!"), Some(3));
        }
    }

    #[test]
    fn rfind_returns_none_for_an_empty_haystack() {
        let all_bytes: Vec<u8> = (0..=255).collect();
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            for needle in [b"".as_slice(), b"a", all_bytes.as_slice()] {
                let matcher = TruffleEngine::new(backend, needle).unwrap();
                assert_eq!(matcher.rfind(b""), None, "needle={needle:?}");
            }
        }
    }

    #[test]
    fn rfind_returns_none_for_an_empty_needle() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            let matcher = TruffleEngine::new(backend, b"").unwrap();
            assert_eq!(matcher.rfind(b"abc"), None);
            // Also search all byte values, spanning multiple SIMD blocks.
            let all_bytes: Vec<u8> = (0..=255).collect();
            assert_eq!(matcher.rfind(&all_bytes), None);
        }
    }

    #[test]
    fn rfind_returns_last_position_for_a_full_needle() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            let needle: Vec<u8> = (0..=255).collect();
            let matcher = TruffleEngine::new(backend, &needle).unwrap();
            assert_eq!(matcher.rfind(b"abc"), Some(2));
            // Also search all byte values, spanning multiple SIMD blocks.
            assert_eq!(matcher.rfind(&needle), Some(255));
        }
    }

    #[test]
    fn rfind_ignores_matches_outside_the_slice() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            let matcher = TruffleEngine::new(backend, &[0]).unwrap();
            // Cover short inputs, vector tails, and every 64-byte alignment.
            for len in 0..=257 {
                for offset in 0..64 {
                    // Only the surrounding zeros match the needle.
                    let mut storage = vec![0; offset + len + 64];
                    let haystack = &mut storage[offset..offset + len];
                    haystack.fill(1);
                    assert_eq!(matcher.rfind(haystack), None, "len={len}, offset={offset}");
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn rfind_stays_inside_guarded_slices() {
        use crate::tests::GuardedPage;

        let mut guarded = GuardedPage::new();
        let page = guarded.as_mut_slice();
        // No members: the search must read through the whole slice.
        page.fill(1);
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            let matcher = TruffleEngine::new(backend, &[0]).unwrap();
            for len in 0..=257 {
                // Start immediately after the leading guard page.
                assert_eq!(matcher.rfind(&page[..len]), None, "start, len={len}");
                // End immediately before the trailing guard page.
                assert_eq!(
                    matcher.rfind(&page[page.len() - len..]),
                    None,
                    "end, len={len}"
                );
            }
        }
    }

    #[test]
    fn rfind_returns_expected_positions() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            for (needle, haystack) in hyperscan_needle_and_haystack() {
                let matcher = TruffleEngine::new(backend, &needle).unwrap();
                let expected = haystack.iter().rposition(|byte| needle.contains(byte));
                // Exercise every start alignment of the widest (64-byte) vector.
                let mut storage = vec![0; haystack.len() + 63];
                for offset in 0..64 {
                    let input = &mut storage[offset..offset + haystack.len()];
                    input.copy_from_slice(&haystack);
                    assert_eq!(
                        matcher.rfind(input),
                        expected,
                        "backend={backend:?}, needle={needle:?}, haystack={haystack:?}, offset={offset}"
                    );
                }
            }
        }
    }

    #[test]
    fn rmismatch_returns_the_last_nonmember() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            let matcher = TruffleEngine::new(backend, b"Ab").unwrap();
            // Both positions qualify; the later one must win.
            let mut haystack = [b'A'; 128];
            haystack[17] = b'!';
            haystack[96] = b'?';

            assert_eq!(matcher.rmismatch(&haystack), Some(96));
        }
    }

    #[test]
    fn rmismatch_recognizes_every_byte_value() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            for member in 0..=255u8 {
                let matcher = TruffleEngine::new(backend, &[member]).unwrap();
                let mut haystack = [member; 129];
                // The only nonmember lies just beyond a 64-byte block.
                haystack[65] = member.wrapping_add(1);
                assert_eq!(matcher.rmismatch(&haystack), Some(65), "member={member}");
            }
        }
    }

    #[test]
    fn rmismatch_distinguishes_members_of_a_sparse_set() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            // h << 4 fills the high nibble; adding h fills the low nibble.
            // These isolated members span the whole byte range.
            let needle: Vec<u8> = (0..16).map(|h| (h << 4) + h).collect();
            let matcher = TruffleEngine::new(backend, &needle).unwrap();
            for byte in 0..=255 {
                let expected = (!needle.contains(&byte)).then_some(127);
                // A uniform input either qualifies everywhere or nowhere.
                assert_eq!(matcher.rmismatch(&[byte; 128]), expected, "byte={byte}");
            }
        }
    }

    #[test]
    fn rmismatch_distinguishes_nonmembers_of_a_dense_set() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            // Exclude bytes whose high and low nibbles are equal.
            let needle: Vec<u8> = (0..=255u8)
                .filter(|byte| byte >> 4 != byte & 0b00001111)
                .collect();
            let matcher = TruffleEngine::new(backend, &needle).unwrap();
            for byte in 0..=255 {
                let expected = (!needle.contains(&byte)).then_some(127);
                // A uniform input either qualifies everywhere or nowhere.
                assert_eq!(matcher.rmismatch(&[byte; 128]), expected, "byte={byte}");
            }
        }
    }

    #[test]
    fn rmismatch_uses_ascii_case_expansion() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            let matcher = TruffleEngine::with_case(true, backend, b"A").unwrap();
            assert_eq!(matcher.rmismatch(b"?Aa!Aa"), Some(3));
        }
    }

    #[test]
    fn rmismatch_returns_none_for_an_empty_haystack() {
        let all_bytes: Vec<u8> = (0..=255).collect();
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            for needle in [b"".as_slice(), b"a", all_bytes.as_slice()] {
                let matcher = TruffleEngine::new(backend, needle).unwrap();
                assert_eq!(matcher.rmismatch(b""), None, "needle={needle:?}");
            }
        }
    }

    #[test]
    fn rmismatch_returns_last_position_for_an_empty_needle() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            let matcher = TruffleEngine::new(backend, b"").unwrap();
            assert_eq!(matcher.rmismatch(b"abc"), Some(2));
            // Also search all byte values, spanning multiple SIMD blocks.
            let all_bytes: Vec<u8> = (0..=255).collect();
            assert_eq!(matcher.rmismatch(&all_bytes), Some(255));
        }
    }

    #[test]
    fn rmismatch_returns_none_for_a_full_needle() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            let needle: Vec<u8> = (0..=255).collect();
            let matcher = TruffleEngine::new(backend, &needle).unwrap();
            assert_eq!(matcher.rmismatch(b"abc"), None);
            // Also search all byte values, spanning multiple SIMD blocks.
            assert_eq!(matcher.rmismatch(&needle), None);
        }
    }

    #[test]
    fn rmismatch_ignores_nonmembers_outside_the_slice() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            let matcher = TruffleEngine::new(backend, &[1]).unwrap();
            // Cover short inputs, vector tails, and every 64-byte alignment.
            for len in 0..=257 {
                for offset in 0..64 {
                    // Only the surrounding zeros are nonmembers.
                    let mut storage = vec![0; offset + len + 64];
                    let haystack = &mut storage[offset..offset + len];
                    haystack.fill(1);
                    assert_eq!(
                        matcher.rmismatch(haystack),
                        None,
                        "len={len}, offset={offset}"
                    );
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn rmismatch_stays_inside_guarded_slices() {
        use crate::tests::GuardedPage;

        let mut guarded = GuardedPage::new();
        let page = guarded.as_mut_slice();
        // No nonmembers: the search must read through the whole slice.
        page.fill(1);
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            let matcher = TruffleEngine::new(backend, &[1]).unwrap();
            for len in 0..=257 {
                // Start immediately after the leading guard page.
                assert_eq!(matcher.rmismatch(&page[..len]), None, "start, len={len}");
                // End immediately before the trailing guard page.
                assert_eq!(
                    matcher.rmismatch(&page[page.len() - len..]),
                    None,
                    "end, len={len}"
                );
            }
        }
    }

    #[test]
    fn rmismatch_returns_expected_positions() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            for (needle, haystack) in hyperscan_needle_and_haystack() {
                let matcher = TruffleEngine::new(backend, &needle).unwrap();
                let expected = haystack.iter().rposition(|byte| !needle.contains(byte));
                // Exercise every start alignment of the widest (64-byte) vector.
                let mut storage = vec![0; haystack.len() + 63];
                for offset in 0..64 {
                    let input = &mut storage[offset..offset + haystack.len()];
                    input.copy_from_slice(&haystack);
                    assert_eq!(
                        matcher.rmismatch(input),
                        expected,
                        "backend={backend:?}, needle={needle:?}, haystack={haystack:?}, offset={offset}"
                    );
                }
            }
        }
    }

    #[test]
    fn finder_uses_the_engines_needle() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            let matcher = TruffleEngine::new(backend, b"Ab").unwrap();
            assert!(matcher.finder(b"!A?b!").eq([1, 3]));
        }
    }

    #[test]
    fn mismatcher_uses_the_engines_needle() {
        for backend in SimdBackend::backends().filter(SimdBackend::is_available) {
            let matcher = TruffleEngine::new(backend, b"Ab").unwrap();
            assert!(matcher.mismatcher(b"!A?b!").eq([0, 2, 4]));
        }
    }
}
