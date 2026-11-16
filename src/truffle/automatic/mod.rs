#[cfg(target_has_atomic = "ptr")]
use core::sync::atomic::{AtomicPtr, Ordering};

mod iter;

pub use self::iter::{TruffleFinder, TruffleMismatcher};

use super::{arch, TruffleTable};
use crate::SimdBackend;

/// A matcher for a set of bytes.
///
/// # Examples
///
/// ```
/// use bytesetscan::truffle::Truffle;
///
/// let delimiters = Truffle::new(b",;");
/// assert_eq!(delimiters.find(b"one;two,three"), Some(3));
/// assert_eq!(delimiters.rfind(b"one;two,three"), Some(7));
/// ```
#[derive(Debug, Clone)]
pub struct Truffle {
    table: TruffleTable,
}

impl Truffle {
    /// Creates a matcher for the bytes in `needle`.
    ///
    /// An empty needle matches no bytes. Duplicate bytes and their order do not
    /// affect matching.
    pub fn new(needle: &[u8]) -> Self {
        Self {
            table: TruffleTable::new(needle),
        }
    }

    /// Creates a matcher that can ignore ASCII letter case.
    ///
    /// When `ignore_case` is `true`, ASCII letters match either case; other
    /// bytes match exactly. When it is `false`, this behaves like [`Self::new`].
    pub fn with_case(ignore_case: bool, needle: &[u8]) -> Self {
        Self {
            table: TruffleTable::with_case(ignore_case, needle),
        }
    }

    /// Returns the index of the first byte in `haystack` that belongs to the needle.
    ///
    /// Returns `None` if no match is found.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytesetscan::truffle::Truffle;
    ///
    /// let whitespace = Truffle::new(b" \t\r\n");
    /// assert_eq!(whitespace.find(b"hello world"), Some(5));
    /// assert_eq!(whitespace.find(b"hello"), None);
    /// ```
    #[inline]
    pub fn find(&self, haystack: &[u8]) -> Option<usize> {
        self.search::<false>(haystack)
    }

    /// Returns the index of the first byte in `haystack` that is not in the needle.
    ///
    /// Returns `None` if all bytes belong to the needle.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytesetscan::truffle::Truffle;
    ///
    /// let bases = Truffle::with_case(true, b"ACGTU");
    /// assert_eq!(bases.mismatch(b"acGU?"), Some(4));
    /// assert_eq!(bases.mismatch(b"acGU"), None);
    /// ```
    #[inline]
    pub fn mismatch(&self, haystack: &[u8]) -> Option<usize> {
        self.search::<true>(haystack)
    }

    // Automatic searches cache kernel pointers whenever the target supports
    // pointer-sized atomics, independently of std and the target architecture.
    // SimdBackend handles availability: std builds can use runtime checks, while
    // no_std builds use compilation target features. Both choices can be cached.
    #[inline(always)]
    fn search<const REVERT: bool>(&self, haystack: &[u8]) -> Option<usize> {
        let lower = self.table.lower();
        let upper = self.table.upper();
        #[cfg(target_has_atomic = "ptr")]
        {
            // Generic instantiations share local statics, so find and mismatch
            // still need separate caches. Only function pointers are published;
            // Relaxed ordering suffices because no other mutable state is shared.
            static FIND: AtomicPtr<()> = AtomicPtr::new(detect::<false> as *mut ());
            static MISMATCH: AtomicPtr<()> = AtomicPtr::new(detect::<true> as *mut ());

            #[cold]
            fn detect<const REVERT: bool>(
                lower: &[u8; 16],
                upper: &[u8; 16],
                haystack: &[u8],
            ) -> usize {
                let search = arch::search_fn::<REVERT>(SimdBackend::new());
                let caching = if REVERT { &MISMATCH } else { &FIND };
                caching.store(search as *mut (), Ordering::Relaxed);
                // SAFETY: SimdBackend::new selected a supported backend.
                // Each entry point handles all slice lengths.
                unsafe { search(lower, upper, haystack) }
            }

            let cached = if REVERT { &MISMATCH } else { &FIND };
            // SAFETY: each cached pointer has the SearchFn signature and is
            // either the initializer or a kernel selected for a supported backend.
            // Every entry point handles its own short-input path.
            let index = unsafe {
                let search: arch::SearchFn = core::mem::transmute(cached.load(Ordering::Relaxed));
                search(lower, upper, haystack)
            };
            (index != haystack.len()).then_some(index)
        }
        #[cfg(not(target_has_atomic = "ptr"))]
        {
            // AtomicPtr is unavailable on this target. Select a supported kernel
            // for this call without publishing a shared function pointer.
            let search = arch::search_fn::<REVERT>(SimdBackend::new());
            // SAFETY: SimdBackend::new selected a supported backend; its entry
            // point handles all slice lengths, including empty slices.
            let index = unsafe { search(lower, upper, haystack) };
            (index != haystack.len()).then_some(index)
        }
    }

    /// Returns the index of the last byte in `haystack` that belongs to the needle.
    ///
    /// Searches from the end of the slice. Returns `None` if no match is found.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytesetscan::truffle::Truffle;
    ///
    /// let whitespace = Truffle::new(b" \t");
    /// assert_eq!(whitespace.rfind(b"one two\tthree"), Some(7));
    /// assert_eq!(whitespace.rfind(b"word"), None);
    /// ```
    #[inline]
    pub fn rfind(&self, haystack: &[u8]) -> Option<usize> {
        self.rsearch::<false>(haystack)
    }

    /// Returns the index of the last byte in `haystack` that is not in the needle.
    ///
    /// Searches from the end of the slice. Returns `None` if all bytes belong
    /// to the needle.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytesetscan::truffle::Truffle;
    ///
    /// let whitespace = Truffle::new(b" \t");
    /// assert_eq!(whitespace.rmismatch(b"one two \t"), Some(6));
    /// assert_eq!(whitespace.rmismatch(b" \t"), None);
    /// ```
    #[inline]
    pub fn rmismatch(&self, haystack: &[u8]) -> Option<usize> {
        self.rsearch::<true>(haystack)
    }

    #[inline(always)]
    fn rsearch<const REVERT: bool>(&self, haystack: &[u8]) -> Option<usize> {
        let lower = self.table.lower();
        let upper = self.table.upper();
        #[cfg(target_has_atomic = "ptr")]
        {
            // Keep the two operations separate even inside this generic function.
            // Concurrent initializations publish the same supported kernel, and
            // SimdBackend::new supplies the choice. Only the pointer needs publishing.
            static RFIND: AtomicPtr<()> = AtomicPtr::new(detect::<false> as *mut ());
            static RMISMATCH: AtomicPtr<()> = AtomicPtr::new(detect::<true> as *mut ());

            #[cold]
            fn detect<const REVERT: bool>(
                lower: &[u8; 16],
                upper: &[u8; 16],
                haystack: &[u8],
            ) -> usize {
                let search = arch::rsearch_fn::<REVERT>(SimdBackend::new());
                let caching = if REVERT { &RMISMATCH } else { &RFIND };
                caching.store(search as *mut (), Ordering::Relaxed);
                // SAFETY: SimdBackend::new selected a supported backend.
                // Each entry point handles all slice lengths.
                unsafe { search(lower, upper, haystack) }
            }

            let cached = if REVERT { &RMISMATCH } else { &RFIND };
            // SAFETY: each cached pointer has the SearchFn signature and is
            // either the initializer or a kernel selected for a supported backend.
            // Every entry point handles its own short-input path.
            let index = unsafe {
                let search: arch::SearchFn = core::mem::transmute(cached.load(Ordering::Relaxed));
                search(lower, upper, haystack)
            };
            (index != haystack.len()).then_some(index)
        }
        #[cfg(not(target_has_atomic = "ptr"))]
        {
            // Use the same selection on targets without pointer-sized atomics,
            // but keep the function pointer local instead of caching it.
            let search = arch::rsearch_fn::<REVERT>(SimdBackend::new());
            // SAFETY: SimdBackend::new selected a supported backend; its entry
            // point handles all slice lengths, including empty slices.
            let index = unsafe { search(lower, upper, haystack) };
            (index != haystack.len()).then_some(index)
        }
    }

    /// Returns an iterator over the indices of bytes belonging to the needle.
    ///
    /// Indices are yielded in ascending order. Use [`.rev()`](Iterator::rev)
    /// to iterate in reverse.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytesetscan::truffle::Truffle;
    ///
    /// let letters = Truffle::new(b"a");
    /// let mut matches = letters.finder(b"banana");
    /// assert_eq!(matches.next(), Some(1));
    /// assert_eq!(matches.next_back(), Some(5));
    /// assert!(matches.eq([3]));
    /// ```
    #[inline]
    pub fn finder<'a, 'h>(&'a self, haystack: &'h [u8]) -> TruffleFinder<'a, 'h> {
        TruffleFinder::new(self, haystack)
    }

    /// Returns an iterator over the indices of bytes not belonging to the needle.
    ///
    /// Indices are yielded in ascending order. Use [`.rev()`](Iterator::rev)
    /// to iterate in reverse.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytesetscan::truffle::Truffle;
    ///
    /// let bases = Truffle::new(b"ACGT");
    /// assert!(bases.mismatcher(b"AC?GT!").eq([2, 5]));
    /// assert!(bases.mismatcher(b"AC?GT!").rev().eq([5, 2]));
    /// ```
    #[inline]
    pub fn mismatcher<'a, 'h>(&'a self, haystack: &'h [u8]) -> TruffleMismatcher<'a, 'h> {
        TruffleMismatcher::new(self, haystack)
    }

    /// Returns an iterator over the needle in ascending byte order.
    ///
    /// This is the matching byte set, not the original constructor input.
    /// Each byte is yielded once. With ASCII case-insensitive matching, both
    /// cases of each letter are included.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytesetscan::truffle::Truffle;
    ///
    /// let letters = Truffle::with_case(true, b"aAa");
    /// assert!(letters.needle().eq(b"Aa".iter().copied()));
    /// ```
    pub fn needle(&self) -> impl Iterator<Item = u8> + '_ {
        self.table.needle()
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::{vec, vec::Vec};

    use super::Truffle;
    use crate::tests::hyperscan_needle_and_haystack;

    #[test]
    fn find_returns_the_first_member() {
        let matcher = Truffle::new(b"Ab");
        // Both positions qualify; the earlier one must win.
        let mut haystack = [b'!'; 128];
        haystack[17] = b'A';
        haystack[96] = b'b';

        assert_eq!(matcher.find(&haystack), Some(17));
    }

    #[test]
    fn find_recognizes_every_byte_value() {
        let haystack: [u8; 256] = core::array::from_fn(|index| index as u8);
        for member in 0..=255u8 {
            let matcher = Truffle::new(&[member]);
            // Each byte occurs once, at the index equal to its value.
            assert_eq!(
                matcher.find(&haystack),
                Some(usize::from(member)),
                "member={member}"
            );
        }
    }

    #[test]
    fn find_distinguishes_members_of_a_sparse_set() {
        // h << 4 fills the high nibble; adding h fills the low nibble.
        // These isolated members span the whole byte range.
        let needle: Vec<u8> = (0..16).map(|h| (h << 4) + h).collect();
        let matcher = Truffle::new(&needle);
        for byte in 0..=255 {
            let expected = (needle.contains(&byte)).then_some(0);
            // A uniform input either qualifies everywhere or nowhere.
            assert_eq!(matcher.find(&[byte; 128]), expected, "byte={byte}");
        }
    }

    #[test]
    fn find_distinguishes_nonmembers_of_a_dense_set() {
        // Exclude bytes whose high and low nibbles are equal.
        let needle: Vec<u8> = (0..=255u8)
            .filter(|byte| byte >> 4 != byte & 0b00001111)
            .collect();
        let matcher = Truffle::new(&needle);
        for byte in 0..=255 {
            let expected = (needle.contains(&byte)).then_some(0);
            // A uniform input either qualifies everywhere or nowhere.
            assert_eq!(matcher.find(&[byte; 128]), expected, "byte={byte}");
        }
    }

    #[test]
    fn find_uses_ascii_case_expansion() {
        let matcher = Truffle::with_case(true, b"A");
        assert_eq!(matcher.find(b"!a!A!"), Some(1));
    }

    #[test]
    fn find_returns_none_for_an_empty_haystack() {
        let all_bytes: Vec<u8> = (0..=255).collect();
        for needle in [b"".as_slice(), b"a", all_bytes.as_slice()] {
            let matcher = Truffle::new(needle);
            assert_eq!(matcher.find(b""), None, "needle={needle:?}");
        }
    }

    #[test]
    fn find_returns_none_for_an_empty_needle() {
        let matcher = Truffle::new(b"");
        assert_eq!(matcher.find(b"abc"), None);
        // Also search all byte values, spanning multiple SIMD blocks.
        let all_bytes: Vec<u8> = (0..=255).collect();
        assert_eq!(matcher.find(&all_bytes), None);
    }

    #[test]
    fn find_returns_first_position_for_a_full_needle() {
        let needle: Vec<u8> = (0..=255).collect();
        let matcher = Truffle::new(&needle);
        assert_eq!(matcher.find(b"abc"), Some(0));
        // Also search all byte values, spanning multiple SIMD blocks.
        assert_eq!(matcher.find(&needle), Some(0));
    }

    #[test]
    fn find_ignores_matches_outside_the_slice() {
        let matcher = Truffle::new(&[0]);
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

    #[cfg(target_os = "linux")]
    #[test]
    fn find_stays_inside_guarded_slices() {
        use crate::tests::GuardedPage;

        let mut guarded = GuardedPage::new();
        let page = guarded.as_mut_slice();
        // No members: the search must read through the whole slice.
        page.fill(1);
        let matcher = Truffle::new(&[0]);
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

    #[test]
    fn find_returns_expected_positions() {
        for (needle, haystack) in hyperscan_needle_and_haystack() {
            let matcher = Truffle::new(&needle);
            let expected = haystack.iter().position(|byte| needle.contains(byte));
            // Exercise every start alignment of the widest (64-byte) vector.
            let mut storage = vec![0; haystack.len() + 63];
            for offset in 0..64 {
                let input = &mut storage[offset..offset + haystack.len()];
                input.copy_from_slice(&haystack);
                assert_eq!(
                    matcher.find(input),
                    expected,
                    "needle={needle:?}, haystack={haystack:?}, offset={offset}"
                );
            }
        }
    }

    #[test]
    fn mismatch_returns_the_first_nonmember() {
        let matcher = Truffle::new(b"Ab");
        // Both positions qualify; the earlier one must win.
        let mut haystack = [b'A'; 128];
        haystack[17] = b'!';
        haystack[96] = b'?';

        assert_eq!(matcher.mismatch(&haystack), Some(17));
    }

    #[test]
    fn mismatch_recognizes_every_byte_value() {
        for member in 0..=255u8 {
            let matcher = Truffle::new(&[member]);
            let mut haystack = [member; 129];
            // The only nonmember lies just beyond a 64-byte block.
            haystack[65] = member.wrapping_add(1);
            assert_eq!(matcher.mismatch(&haystack), Some(65), "member={member}");
        }
    }

    #[test]
    fn mismatch_distinguishes_members_of_a_sparse_set() {
        // h << 4 fills the high nibble; adding h fills the low nibble.
        // These isolated members span the whole byte range.
        let needle: Vec<u8> = (0..16).map(|h| (h << 4) + h).collect();
        let matcher = Truffle::new(&needle);
        for byte in 0..=255 {
            let expected = (!needle.contains(&byte)).then_some(0);
            // A uniform input either qualifies everywhere or nowhere.
            assert_eq!(matcher.mismatch(&[byte; 128]), expected, "byte={byte}");
        }
    }

    #[test]
    fn mismatch_distinguishes_nonmembers_of_a_dense_set() {
        // Exclude bytes whose high and low nibbles are equal.
        let needle: Vec<u8> = (0..=255u8)
            .filter(|byte| byte >> 4 != byte & 0b00001111)
            .collect();
        let matcher = Truffle::new(&needle);
        for byte in 0..=255 {
            let expected = (!needle.contains(&byte)).then_some(0);
            // A uniform input either qualifies everywhere or nowhere.
            assert_eq!(matcher.mismatch(&[byte; 128]), expected, "byte={byte}");
        }
    }

    #[test]
    fn mismatch_uses_ascii_case_expansion() {
        let matcher = Truffle::with_case(true, b"A");
        assert_eq!(matcher.mismatch(b"aA!aA?"), Some(2));
    }

    #[test]
    fn mismatch_returns_none_for_an_empty_haystack() {
        let all_bytes: Vec<u8> = (0..=255).collect();
        for needle in [b"".as_slice(), b"a", all_bytes.as_slice()] {
            let matcher = Truffle::new(needle);
            assert_eq!(matcher.mismatch(b""), None, "needle={needle:?}");
        }
    }

    #[test]
    fn mismatch_returns_first_position_for_an_empty_needle() {
        let matcher = Truffle::new(b"");
        assert_eq!(matcher.mismatch(b"abc"), Some(0));
        // Also search all byte values, spanning multiple SIMD blocks.
        let all_bytes: Vec<u8> = (0..=255).collect();
        assert_eq!(matcher.mismatch(&all_bytes), Some(0));
    }

    #[test]
    fn mismatch_returns_none_for_a_full_needle() {
        let needle: Vec<u8> = (0..=255).collect();
        let matcher = Truffle::new(&needle);
        assert_eq!(matcher.mismatch(b"abc"), None);
        // Also search all byte values, spanning multiple SIMD blocks.
        assert_eq!(matcher.mismatch(&needle), None);
    }

    #[test]
    fn mismatch_ignores_nonmembers_outside_the_slice() {
        let matcher = Truffle::new(&[1]);
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

    #[cfg(target_os = "linux")]
    #[test]
    fn mismatch_stays_inside_guarded_slices() {
        use crate::tests::GuardedPage;

        let mut guarded = GuardedPage::new();
        let page = guarded.as_mut_slice();
        // No nonmembers: the search must read through the whole slice.
        page.fill(1);
        let matcher = Truffle::new(&[1]);
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

    #[test]
    fn mismatch_returns_expected_positions() {
        for (needle, haystack) in hyperscan_needle_and_haystack() {
            let matcher = Truffle::new(&needle);
            let expected = haystack.iter().position(|byte| !needle.contains(byte));
            // Exercise every start alignment of the widest (64-byte) vector.
            let mut storage = vec![0; haystack.len() + 63];
            for offset in 0..64 {
                let input = &mut storage[offset..offset + haystack.len()];
                input.copy_from_slice(&haystack);
                assert_eq!(
                    matcher.mismatch(input),
                    expected,
                    "needle={needle:?}, haystack={haystack:?}, offset={offset}"
                );
            }
        }
    }

    #[test]
    fn rfind_returns_the_last_member() {
        let matcher = Truffle::new(b"Ab");
        // Both positions qualify; the later one must win.
        let mut haystack = [b'!'; 128];
        haystack[17] = b'A';
        haystack[96] = b'b';

        assert_eq!(matcher.rfind(&haystack), Some(96));
    }

    #[test]
    fn rfind_recognizes_every_byte_value() {
        let haystack: [u8; 256] = core::array::from_fn(|index| index as u8);
        for member in 0..=255u8 {
            let matcher = Truffle::new(&[member]);
            // Each byte occurs once, at the index equal to its value.
            assert_eq!(
                matcher.rfind(&haystack),
                Some(usize::from(member)),
                "member={member}"
            );
        }
    }

    #[test]
    fn rfind_distinguishes_members_of_a_sparse_set() {
        // h << 4 fills the high nibble; adding h fills the low nibble.
        // These isolated members span the whole byte range.
        let needle: Vec<u8> = (0..16).map(|h| (h << 4) + h).collect();
        let matcher = Truffle::new(&needle);
        for byte in 0..=255 {
            let expected = (needle.contains(&byte)).then_some(127);
            // A uniform input either qualifies everywhere or nowhere.
            assert_eq!(matcher.rfind(&[byte; 128]), expected, "byte={byte}");
        }
    }

    #[test]
    fn rfind_distinguishes_nonmembers_of_a_dense_set() {
        // Exclude bytes whose high and low nibbles are equal.
        let needle: Vec<u8> = (0..=255u8)
            .filter(|byte| byte >> 4 != byte & 0b00001111)
            .collect();
        let matcher = Truffle::new(&needle);
        for byte in 0..=255 {
            let expected = (needle.contains(&byte)).then_some(127);
            // A uniform input either qualifies everywhere or nowhere.
            assert_eq!(matcher.rfind(&[byte; 128]), expected, "byte={byte}");
        }
    }

    #[test]
    fn rfind_uses_ascii_case_expansion() {
        let matcher = Truffle::with_case(true, b"A");
        assert_eq!(matcher.rfind(b"!A!a!"), Some(3));
    }

    #[test]
    fn rfind_returns_none_for_an_empty_haystack() {
        let all_bytes: Vec<u8> = (0..=255).collect();
        for needle in [b"".as_slice(), b"a", all_bytes.as_slice()] {
            let matcher = Truffle::new(needle);
            assert_eq!(matcher.rfind(b""), None, "needle={needle:?}");
        }
    }

    #[test]
    fn rfind_returns_none_for_an_empty_needle() {
        let matcher = Truffle::new(b"");
        assert_eq!(matcher.rfind(b"abc"), None);
        // Also search all byte values, spanning multiple SIMD blocks.
        let all_bytes: Vec<u8> = (0..=255).collect();
        assert_eq!(matcher.rfind(&all_bytes), None);
    }

    #[test]
    fn rfind_returns_last_position_for_a_full_needle() {
        let needle: Vec<u8> = (0..=255).collect();
        let matcher = Truffle::new(&needle);
        assert_eq!(matcher.rfind(b"abc"), Some(2));
        // Also search all byte values, spanning multiple SIMD blocks.
        assert_eq!(matcher.rfind(&needle), Some(255));
    }

    #[test]
    fn rfind_ignores_matches_outside_the_slice() {
        let matcher = Truffle::new(&[0]);
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

    #[cfg(target_os = "linux")]
    #[test]
    fn rfind_stays_inside_guarded_slices() {
        use crate::tests::GuardedPage;

        let mut guarded = GuardedPage::new();
        let page = guarded.as_mut_slice();
        // No members: the search must read through the whole slice.
        page.fill(1);
        let matcher = Truffle::new(&[0]);
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

    #[test]
    fn rfind_returns_expected_positions() {
        for (needle, haystack) in hyperscan_needle_and_haystack() {
            let matcher = Truffle::new(&needle);
            let expected = haystack.iter().rposition(|byte| needle.contains(byte));
            // Exercise every start alignment of the widest (64-byte) vector.
            let mut storage = vec![0; haystack.len() + 63];
            for offset in 0..64 {
                let input = &mut storage[offset..offset + haystack.len()];
                input.copy_from_slice(&haystack);
                assert_eq!(
                    matcher.rfind(input),
                    expected,
                    "needle={needle:?}, haystack={haystack:?}, offset={offset}"
                );
            }
        }
    }

    #[test]
    fn rmismatch_returns_the_last_nonmember() {
        let matcher = Truffle::new(b"Ab");
        // Both positions qualify; the later one must win.
        let mut haystack = [b'A'; 128];
        haystack[17] = b'!';
        haystack[96] = b'?';

        assert_eq!(matcher.rmismatch(&haystack), Some(96));
    }

    #[test]
    fn rmismatch_recognizes_every_byte_value() {
        for member in 0..=255u8 {
            let matcher = Truffle::new(&[member]);
            let mut haystack = [member; 129];
            // The only nonmember lies just beyond a 64-byte block.
            haystack[65] = member.wrapping_add(1);
            assert_eq!(matcher.rmismatch(&haystack), Some(65), "member={member}");
        }
    }

    #[test]
    fn rmismatch_distinguishes_members_of_a_sparse_set() {
        // h << 4 fills the high nibble; adding h fills the low nibble.
        // These isolated members span the whole byte range.
        let needle: Vec<u8> = (0..16).map(|h| (h << 4) + h).collect();
        let matcher = Truffle::new(&needle);
        for byte in 0..=255 {
            let expected = (!needle.contains(&byte)).then_some(127);
            // A uniform input either qualifies everywhere or nowhere.
            assert_eq!(matcher.rmismatch(&[byte; 128]), expected, "byte={byte}");
        }
    }

    #[test]
    fn rmismatch_distinguishes_nonmembers_of_a_dense_set() {
        // Exclude bytes whose high and low nibbles are equal.
        let needle: Vec<u8> = (0..=255u8)
            .filter(|byte| byte >> 4 != byte & 0b00001111)
            .collect();
        let matcher = Truffle::new(&needle);
        for byte in 0..=255 {
            let expected = (!needle.contains(&byte)).then_some(127);
            // A uniform input either qualifies everywhere or nowhere.
            assert_eq!(matcher.rmismatch(&[byte; 128]), expected, "byte={byte}");
        }
    }

    #[test]
    fn rmismatch_uses_ascii_case_expansion() {
        let matcher = Truffle::with_case(true, b"A");
        assert_eq!(matcher.rmismatch(b"?Aa!Aa"), Some(3));
    }

    #[test]
    fn rmismatch_returns_none_for_an_empty_haystack() {
        let all_bytes: Vec<u8> = (0..=255).collect();
        for needle in [b"".as_slice(), b"a", all_bytes.as_slice()] {
            let matcher = Truffle::new(needle);
            assert_eq!(matcher.rmismatch(b""), None, "needle={needle:?}");
        }
    }

    #[test]
    fn rmismatch_returns_last_position_for_an_empty_needle() {
        let matcher = Truffle::new(b"");
        assert_eq!(matcher.rmismatch(b"abc"), Some(2));
        // Also search all byte values, spanning multiple SIMD blocks.
        let all_bytes: Vec<u8> = (0..=255).collect();
        assert_eq!(matcher.rmismatch(&all_bytes), Some(255));
    }

    #[test]
    fn rmismatch_returns_none_for_a_full_needle() {
        let needle: Vec<u8> = (0..=255).collect();
        let matcher = Truffle::new(&needle);
        assert_eq!(matcher.rmismatch(b"abc"), None);
        // Also search all byte values, spanning multiple SIMD blocks.
        assert_eq!(matcher.rmismatch(&needle), None);
    }

    #[test]
    fn rmismatch_ignores_nonmembers_outside_the_slice() {
        let matcher = Truffle::new(&[1]);
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

    #[cfg(target_os = "linux")]
    #[test]
    fn rmismatch_stays_inside_guarded_slices() {
        use crate::tests::GuardedPage;

        let mut guarded = GuardedPage::new();
        let page = guarded.as_mut_slice();
        // No nonmembers: the search must read through the whole slice.
        page.fill(1);
        let matcher = Truffle::new(&[1]);
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

    #[test]
    fn rmismatch_returns_expected_positions() {
        for (needle, haystack) in hyperscan_needle_and_haystack() {
            let matcher = Truffle::new(&needle);
            let expected = haystack.iter().rposition(|byte| !needle.contains(byte));
            // Exercise every start alignment of the widest (64-byte) vector.
            let mut storage = vec![0; haystack.len() + 63];
            for offset in 0..64 {
                let input = &mut storage[offset..offset + haystack.len()];
                input.copy_from_slice(&haystack);
                assert_eq!(
                    matcher.rmismatch(input),
                    expected,
                    "needle={needle:?}, haystack={haystack:?}, offset={offset}"
                );
            }
        }
    }

    #[test]
    fn finder_uses_the_matchers_needle() {
        let matcher = Truffle::new(b"Ab");
        assert!(matcher.finder(b"!A?b!").eq([1, 3]));
    }

    #[test]
    fn mismatcher_uses_the_matchers_needle() {
        let matcher = Truffle::new(b"Ab");
        assert!(matcher.mismatcher(b"!A?b!").eq([0, 2, 4]));
    }
}
