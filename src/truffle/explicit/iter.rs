use core::iter::FusedIterator;

use super::TruffleEngine;

/// An iterator over the indices of bytes belonging to the needle.
///
/// Created by [`TruffleEngine::finder`].
pub struct TruffleEngineFinder<'a, 'h> {
    inner: TruffleEngineIter<'a, 'h, false>,
}

/// An iterator over the indices of bytes not belonging to the needle.
///
/// Created by [`TruffleEngine::mismatcher`].
pub struct TruffleEngineMismatcher<'a, 'h> {
    inner: TruffleEngineIter<'a, 'h, true>,
}

impl<'a, 'h> TruffleEngineFinder<'a, 'h> {
    #[inline]
    pub(super) fn new(engine: &'a TruffleEngine, haystack: &'h [u8]) -> Self {
        Self {
            inner: TruffleEngineIter::new(engine, haystack),
        }
    }
}

impl<'a, 'h> TruffleEngineMismatcher<'a, 'h> {
    #[inline]
    pub(super) fn new(engine: &'a TruffleEngine, haystack: &'h [u8]) -> Self {
        Self {
            inner: TruffleEngineIter::new(engine, haystack),
        }
    }
}

impl Iterator for TruffleEngineFinder<'_, '_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl DoubleEndedIterator for TruffleEngineFinder<'_, '_> {
    #[inline]
    fn next_back(&mut self) -> Option<usize> {
        self.inner.next_back()
    }
}

impl FusedIterator for TruffleEngineFinder<'_, '_> {}

impl Iterator for TruffleEngineMismatcher<'_, '_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl DoubleEndedIterator for TruffleEngineMismatcher<'_, '_> {
    #[inline]
    fn next_back(&mut self) -> Option<usize> {
        self.inner.next_back()
    }
}

impl FusedIterator for TruffleEngineMismatcher<'_, '_> {}

// Keep only the unconsumed interval and its engine. Each step searches the
// remaining bytes and excludes the returned byte from subsequent searches.
// Each step calls the borrowed engine's vtable, retaining its selected backend
// throughout iteration without consulting the automatic dispatch caches.
struct TruffleEngineIter<'a, 'h, const REVERT: bool> {
    engine: &'a TruffleEngine,
    haystack: &'h [u8],
    front: usize,
    back: usize,
}

impl<const REVERT: bool> Iterator for TruffleEngineIter<'_, '_, REVERT> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.is_empty() {
            return None;
        }
        match self
            .engine
            .vtable
            .search::<REVERT>(&self.engine.table, self.haystack())
        {
            Some(offset) => {
                // Search returns an index inside the remaining interval, so
                // index + 1 cannot exceed back. Preserve the original offset.
                let index = self.front + offset;
                self.front = index + 1;
                Some(index)
            }
            None => {
                // A failed search exhausts the interval in both directions.
                self.front = self.back;
                None
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.len()))
    }
}

impl<const REVERT: bool> DoubleEndedIterator for TruffleEngineIter<'_, '_, REVERT> {
    #[inline]
    fn next_back(&mut self) -> Option<usize> {
        if self.is_empty() {
            return None;
        }
        match self
            .engine
            .vtable
            .rsearch::<REVERT>(&self.engine.table, self.haystack())
        {
            Some(relative) => {
                // The returned byte becomes the excluded end of the interval.
                // Sharing these bounds with next prevents duplicate results
                // when callers alternate between the two directions.
                let index = self.front + relative;
                self.back = index;
                Some(index)
            }
            None => {
                self.front = self.back;
                None
            }
        }
    }
}

impl<const REVERT: bool> FusedIterator for TruffleEngineIter<'_, '_, REVERT> {}

impl<'a, 'h, const REVERT: bool> TruffleEngineIter<'a, 'h, REVERT> {
    #[inline]
    fn new(engine: &'a TruffleEngine, haystack: &'h [u8]) -> Self {
        Self {
            engine,
            haystack,
            front: 0,
            back: haystack.len(),
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // This counts unsearched bytes, not the number of remaining results.
    #[inline]
    fn len(&self) -> usize {
        self.back - self.front
    }

    #[inline]
    fn haystack(&self) -> &[u8] {
        debug_assert!(self.front <= self.back && self.back <= self.haystack.len());
        // SAFETY: new initializes 0 <= front <= back == haystack.len(). Each
        // successful search returns an index inside this interval: next moves
        // front to at most back, and next_back moves back to at least front.
        // A failed search sets front == back. The fields remain private, so
        // every state transition preserves valid slice bounds.
        unsafe { self.haystack.get_unchecked(self.front..self.back) }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::vec::Vec;

    use super::{TruffleEngine, TruffleEngineFinder, TruffleEngineMismatcher};
    use crate::tests::supported_backends;

    #[test]
    fn finder_next_yields_ascending_indices() {
        for backend in supported_backends() {
            let matcher = TruffleEngine::new(backend, b"A").unwrap();
            // Results on both sides of 64 and 128 exercise several scan blocks.
            let mut haystack = [b'!'; 130];
            for index in [1, 63, 64, 65, 128] {
                haystack[index] = b'A';
            }
            let iter = TruffleEngineFinder::new(&matcher, &haystack);
            assert_eq!(iter.collect::<Vec<_>>(), [1, 63, 64, 65, 128]);
        }
    }

    #[test]
    fn finder_next_stays_exhausted() {
        for backend in supported_backends() {
            let matcher = TruffleEngine::new(backend, b"A").unwrap();
            let mut iter = TruffleEngineFinder::new(&matcher, b"!A!");
            assert_eq!(iter.next(), Some(1));
            for _ in 0..3 {
                assert_eq!(iter.next(), None);
            }
        }
    }

    #[test]
    fn finder_next_back_yields_descending_indices() {
        for backend in supported_backends() {
            let matcher = TruffleEngine::new(backend, b"A").unwrap();
            // Results on both sides of 64 and 128 exercise several scan blocks.
            let mut haystack = [b'!'; 130];
            for index in [1, 63, 64, 65, 128] {
                haystack[index] = b'A';
            }
            let iter = TruffleEngineFinder::new(&matcher, &haystack);
            assert_eq!(iter.rev().collect::<Vec<_>>(), [128, 65, 64, 63, 1]);
        }
    }

    #[test]
    fn finder_next_back_stays_exhausted() {
        for backend in supported_backends() {
            let matcher = TruffleEngine::new(backend, b"A").unwrap();
            let mut iter = TruffleEngineFinder::new(&matcher, b"!A!");
            assert_eq!(iter.next_back(), Some(1));
            for _ in 0..3 {
                assert_eq!(iter.next_back(), None);
            }
        }
    }

    #[test]
    fn finder_size_hint_bounds_the_remaining_results() {
        for backend in supported_backends() {
            let matcher = TruffleEngine::new(backend, b"A").unwrap();
            let mut iter = TruffleEngineFinder::new(&matcher, b"A!A!A");
            // Exactly three results remain initially; each next() consumes one.
            for remaining in (0..=3).rev() {
                let (lower, upper) = iter.size_hint();
                assert!(lower <= remaining);
                assert!(upper.is_some_and(|upper| upper >= remaining));
                iter.next();
            }
        }
    }

    #[test]
    fn finder_size_hint_is_zero_after_exhaustion() {
        for backend in supported_backends() {
            let matcher = TruffleEngine::new(backend, b"A").unwrap();
            let mut iter = TruffleEngineFinder::new(&matcher, b"!A!");
            while iter.next().is_some() {}
            assert_eq!(iter.size_hint(), (0, Some(0)));
        }
    }

    #[test]
    fn finder_next_and_next_back_share_the_remaining_range() {
        for backend in supported_backends() {
            let matcher = TruffleEngine::new(backend, b"A").unwrap();
            // Results on both sides of 64 and 128 exercise several scan blocks.
            let mut haystack = [b'!'; 130];
            for index in [1, 63, 64, 65, 128] {
                haystack[index] = b'A';
            }
            let mut iter = TruffleEngineFinder::new(&matcher, &haystack);
            // Consume from both ends until they meet at the middle result.
            assert_eq!(iter.next(), Some(1));
            assert_eq!(iter.next_back(), Some(128));
            assert_eq!(iter.next(), Some(63));
            assert_eq!(iter.next_back(), Some(65));
            assert_eq!(iter.next(), Some(64));
            assert_eq!(iter.next(), None);
            assert_eq!(iter.next_back(), None);
        }
    }

    #[test]
    fn finder_next_is_exhausted_after_next_back() {
        for backend in supported_backends() {
            let matcher = TruffleEngine::new(backend, b"A").unwrap();
            let mut iter = TruffleEngineFinder::new(&matcher, b"!A!");
            while iter.next_back().is_some() {}
            assert_eq!(iter.next(), None);
        }
    }

    #[test]
    fn finder_returns_no_indices_without_a_match() {
        for backend in supported_backends() {
            let matcher = TruffleEngine::new(backend, b"A").unwrap();
            let iter = TruffleEngineFinder::new(&matcher, b"!!!");
            assert_eq!(iter.collect::<Vec<_>>(), []);
        }
    }

    #[test]
    fn finder_next_yields_every_position_across_block_boundaries() {
        for backend in supported_backends() {
            let matcher = TruffleEngine::new(backend, b"A").unwrap();
            // Include short inputs and lengths just below, at, and above vector boundaries.
            for len in [
                0, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511,
            ] {
                for offset in 0..64 {
                    let mut storage = [b'!'; 576];
                    let haystack = &mut storage[offset..offset + len];
                    haystack.fill(b'A');
                    let expected: Vec<_> = (0..len).collect();
                    let iter = TruffleEngineFinder::new(&matcher, haystack);
                    assert_eq!(
                        iter.collect::<Vec<_>>(),
                        expected,
                        "len={len}, offset={offset}"
                    );
                }
            }
        }
    }

    #[test]
    fn finder_next_back_yields_every_position_across_block_boundaries() {
        for backend in supported_backends() {
            let matcher = TruffleEngine::new(backend, b"A").unwrap();
            // Include short inputs and lengths just below, at, and above vector boundaries.
            for len in [
                0, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511,
            ] {
                for offset in 0..64 {
                    let mut storage = [b'!'; 576];
                    let haystack = &mut storage[offset..offset + len];
                    haystack.fill(b'A');
                    let expected: Vec<_> = (0..len).rev().collect();
                    let iter = TruffleEngineFinder::new(&matcher, haystack);
                    assert_eq!(
                        iter.rev().collect::<Vec<_>>(),
                        expected,
                        "len={len}, offset={offset}"
                    );
                }
            }
        }
    }

    #[test]
    fn finder_next_keeps_spaced_matches_across_block_boundaries() {
        for backend in supported_backends() {
            let matcher = TruffleEngine::new(backend, b"A").unwrap();
            // Include short inputs and lengths just below, at, and above vector boundaries.
            for len in [
                0, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511,
            ] {
                for offset in 0..64 {
                    let mut storage = [b'!'; 576];
                    let haystack = &mut storage[offset..offset + len];
                    for index in (0..len).step_by(3) {
                        haystack[index] = b'A';
                    }
                    let expected: Vec<_> = (0..len).step_by(3).collect();
                    let iter = TruffleEngineFinder::new(&matcher, haystack);
                    assert_eq!(
                        iter.collect::<Vec<_>>(),
                        expected,
                        "len={len}, offset={offset}"
                    );
                }
            }
        }
    }

    #[test]
    fn finder_next_back_keeps_spaced_matches_across_block_boundaries() {
        for backend in supported_backends() {
            let matcher = TruffleEngine::new(backend, b"A").unwrap();
            // Include short inputs and lengths just below, at, and above vector boundaries.
            for len in [
                0, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511,
            ] {
                for offset in 0..64 {
                    let mut storage = [b'!'; 576];
                    let haystack = &mut storage[offset..offset + len];
                    for index in (0..len).step_by(3) {
                        haystack[index] = b'A';
                    }
                    let expected: Vec<_> = (0..len).step_by(3).rev().collect();
                    let iter = TruffleEngineFinder::new(&matcher, haystack);
                    assert_eq!(
                        iter.rev().collect::<Vec<_>>(),
                        expected,
                        "len={len}, offset={offset}"
                    );
                }
            }
        }
    }

    #[test]
    fn mismatcher_next_yields_ascending_indices() {
        for backend in supported_backends() {
            let matcher = TruffleEngine::new(backend, b"A").unwrap();
            // Results on both sides of 64 and 128 exercise several scan blocks.
            let mut haystack = [b'A'; 130];
            for index in [1, 63, 64, 65, 128] {
                haystack[index] = b'!';
            }
            let iter = TruffleEngineMismatcher::new(&matcher, &haystack);
            assert_eq!(iter.collect::<Vec<_>>(), [1, 63, 64, 65, 128]);
        }
    }

    #[test]
    fn mismatcher_next_stays_exhausted() {
        for backend in supported_backends() {
            let matcher = TruffleEngine::new(backend, b"A").unwrap();
            let mut iter = TruffleEngineMismatcher::new(&matcher, b"A!A");
            assert_eq!(iter.next(), Some(1));
            for _ in 0..3 {
                assert_eq!(iter.next(), None);
            }
        }
    }

    #[test]
    fn mismatcher_next_back_yields_descending_indices() {
        for backend in supported_backends() {
            let matcher = TruffleEngine::new(backend, b"A").unwrap();
            // Results on both sides of 64 and 128 exercise several scan blocks.
            let mut haystack = [b'A'; 130];
            for index in [1, 63, 64, 65, 128] {
                haystack[index] = b'!';
            }
            let iter = TruffleEngineMismatcher::new(&matcher, &haystack);
            assert_eq!(iter.rev().collect::<Vec<_>>(), [128, 65, 64, 63, 1]);
        }
    }

    #[test]
    fn mismatcher_next_back_stays_exhausted() {
        for backend in supported_backends() {
            let matcher = TruffleEngine::new(backend, b"A").unwrap();
            let mut iter = TruffleEngineMismatcher::new(&matcher, b"A!A");
            assert_eq!(iter.next_back(), Some(1));
            for _ in 0..3 {
                assert_eq!(iter.next_back(), None);
            }
        }
    }

    #[test]
    fn mismatcher_size_hint_bounds_the_remaining_results() {
        for backend in supported_backends() {
            let matcher = TruffleEngine::new(backend, b"A").unwrap();
            let mut iter = TruffleEngineMismatcher::new(&matcher, b"!A!A!");
            // Exactly three results remain initially; each next() consumes one.
            for remaining in (0..=3).rev() {
                let (lower, upper) = iter.size_hint();
                assert!(lower <= remaining);
                assert!(upper.is_some_and(|upper| upper >= remaining));
                iter.next();
            }
        }
    }

    #[test]
    fn mismatcher_size_hint_is_zero_after_exhaustion() {
        for backend in supported_backends() {
            let matcher = TruffleEngine::new(backend, b"A").unwrap();
            let mut iter = TruffleEngineMismatcher::new(&matcher, b"A!A");
            while iter.next().is_some() {}
            assert_eq!(iter.size_hint(), (0, Some(0)));
        }
    }

    #[test]
    fn mismatcher_next_and_next_back_share_the_remaining_range() {
        for backend in supported_backends() {
            let matcher = TruffleEngine::new(backend, b"A").unwrap();
            // Results on both sides of 64 and 128 exercise several scan blocks.
            let mut haystack = [b'A'; 130];
            for index in [1, 63, 64, 65, 128] {
                haystack[index] = b'!';
            }
            let mut iter = TruffleEngineMismatcher::new(&matcher, &haystack);
            // Consume from both ends until they meet at the middle result.
            assert_eq!(iter.next(), Some(1));
            assert_eq!(iter.next_back(), Some(128));
            assert_eq!(iter.next(), Some(63));
            assert_eq!(iter.next_back(), Some(65));
            assert_eq!(iter.next(), Some(64));
            assert_eq!(iter.next(), None);
            assert_eq!(iter.next_back(), None);
        }
    }

    #[test]
    fn mismatcher_next_is_exhausted_after_next_back() {
        for backend in supported_backends() {
            let matcher = TruffleEngine::new(backend, b"A").unwrap();
            let mut iter = TruffleEngineMismatcher::new(&matcher, b"A!A");
            while iter.next_back().is_some() {}
            assert_eq!(iter.next(), None);
        }
    }

    #[test]
    fn mismatcher_returns_no_indices_without_a_match() {
        for backend in supported_backends() {
            let matcher = TruffleEngine::new(backend, b"A").unwrap();
            let iter = TruffleEngineMismatcher::new(&matcher, b"AAA");
            assert_eq!(iter.collect::<Vec<_>>(), []);
        }
    }

    #[test]
    fn mismatcher_next_yields_every_position_across_block_boundaries() {
        for backend in supported_backends() {
            let matcher = TruffleEngine::new(backend, b"A").unwrap();
            // Include short inputs and lengths just below, at, and above vector boundaries.
            for len in [
                0, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511,
            ] {
                for offset in 0..64 {
                    let mut storage = [b'A'; 576];
                    let haystack = &mut storage[offset..offset + len];
                    haystack.fill(b'!');
                    let expected: Vec<_> = (0..len).collect();
                    let iter = TruffleEngineMismatcher::new(&matcher, haystack);
                    assert_eq!(
                        iter.collect::<Vec<_>>(),
                        expected,
                        "len={len}, offset={offset}"
                    );
                }
            }
        }
    }

    #[test]
    fn mismatcher_next_back_yields_every_position_across_block_boundaries() {
        for backend in supported_backends() {
            let matcher = TruffleEngine::new(backend, b"A").unwrap();
            // Include short inputs and lengths just below, at, and above vector boundaries.
            for len in [
                0, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511,
            ] {
                for offset in 0..64 {
                    let mut storage = [b'A'; 576];
                    let haystack = &mut storage[offset..offset + len];
                    haystack.fill(b'!');
                    let expected: Vec<_> = (0..len).rev().collect();
                    let iter = TruffleEngineMismatcher::new(&matcher, haystack);
                    assert_eq!(
                        iter.rev().collect::<Vec<_>>(),
                        expected,
                        "len={len}, offset={offset}"
                    );
                }
            }
        }
    }

    #[test]
    fn mismatcher_next_keeps_spaced_matches_across_block_boundaries() {
        for backend in supported_backends() {
            let matcher = TruffleEngine::new(backend, b"A").unwrap();
            // Include short inputs and lengths just below, at, and above vector boundaries.
            for len in [
                0, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511,
            ] {
                for offset in 0..64 {
                    let mut storage = [b'A'; 576];
                    let haystack = &mut storage[offset..offset + len];
                    for index in (0..len).step_by(3) {
                        haystack[index] = b'!';
                    }
                    let expected: Vec<_> = (0..len).step_by(3).collect();
                    let iter = TruffleEngineMismatcher::new(&matcher, haystack);
                    assert_eq!(
                        iter.collect::<Vec<_>>(),
                        expected,
                        "len={len}, offset={offset}"
                    );
                }
            }
        }
    }

    #[test]
    fn mismatcher_next_back_keeps_spaced_matches_across_block_boundaries() {
        for backend in supported_backends() {
            let matcher = TruffleEngine::new(backend, b"A").unwrap();
            // Include short inputs and lengths just below, at, and above vector boundaries.
            for len in [
                0, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511,
            ] {
                for offset in 0..64 {
                    let mut storage = [b'A'; 576];
                    let haystack = &mut storage[offset..offset + len];
                    for index in (0..len).step_by(3) {
                        haystack[index] = b'!';
                    }
                    let expected: Vec<_> = (0..len).step_by(3).rev().collect();
                    let iter = TruffleEngineMismatcher::new(&matcher, haystack);
                    assert_eq!(
                        iter.rev().collect::<Vec<_>>(),
                        expected,
                        "len={len}, offset={offset}"
                    );
                }
            }
        }
    }
}
