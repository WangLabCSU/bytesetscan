use super::super::{arch, TruffleTable};
use crate::SimdBackend;

// The four search entry points for one supported backend. Each SearchFn is a
// typed function pointer, not a closure or trait object. The engine stores this
// table inline so searches need neither backend selection nor the automatic
// search's global function-pointer caches.
#[derive(Debug, Clone)]
pub(super) struct VTable {
    find: arch::SearchFn,
    mismatch: arch::SearchFn,
    rfind: arch::SearchFn,
    rmismatch: arch::SearchFn,
}

impl VTable {
    pub(super) fn new(backend: SimdBackend) -> Option<Self> {
        if !backend.is_available() {
            return None;
        }
        // Validate once before retaining any kernel. Every pointer in this
        // table then targets a supported instruction set, including after a
        // clone. Each entry point handles its own short-input path.
        Some(Self {
            find: arch::search_fn::<false>(backend),
            mismatch: arch::search_fn::<true>(backend),
            rfind: arch::rsearch_fn::<false>(backend),
            rmismatch: arch::rsearch_fn::<true>(backend),
        })
    }

    // REVERT selects membership (false) or nonmembership (true) at compile
    // time. Both pointers already belong to the backend validated by new;
    // choosing an operation does not perform another availability check.
    #[inline(always)]
    pub(super) fn search<const REVERT: bool>(
        &self,
        table: &TruffleTable,
        haystack: &[u8],
    ) -> Option<usize> {
        let search = if REVERT { self.mismatch } else { self.find };
        // SAFETY: new validated the stored kernel's instruction set. The table
        // supplies two complete 16-byte masks, and every kernel handles all
        // slice lengths, including empty inputs, without reading outside them.
        let index = unsafe { search(table.lower(), table.upper(), haystack) };
        // Kernels use the slice length as the sentinel for an exhausted search.
        (index != haystack.len()).then_some(index)
    }

    // Reverse scans use their own entry points so the last matching byte is
    // found from the end. REVERT has the same meaning as in search.
    #[inline(always)]
    pub(super) fn rsearch<const REVERT: bool>(
        &self,
        table: &TruffleTable,
        haystack: &[u8],
    ) -> Option<usize> {
        let search = if REVERT { self.rmismatch } else { self.rfind };
        // SAFETY: new validated the stored reverse kernel's instruction set.
        // Both masks are complete, and each reverse entry point handles empty
        // and short inputs as well as full SIMD blocks within the slice bounds.
        let index = unsafe { search(table.lower(), table.upper(), haystack) };
        // Reverse kernels use the same end sentinel as forward kernels.
        (index != haystack.len()).then_some(index)
    }
}

#[cfg(test)]
mod tests {
    use super::{TruffleTable, VTable};
    use crate::tests::{backends, supported_backends};

    #[test]
    fn new_accepts_available_backends() {
        for backend in supported_backends() {
            assert!(VTable::new(backend).is_some(), "{backend:?}");
        }
    }

    #[test]
    fn new_rejects_unavailable_backends() {
        for backend in backends().filter(|b| !b.is_available()) {
            assert!(VTable::new(backend).is_none(), "{backend:?}");
        }
    }

    #[test]
    fn search_dispatches_to_find() {
        let table = TruffleTable::new(b"a");
        let vtable = mock_vtable();
        assert_eq!(vtable.search::<false>(&table, b"abcd"), Some(0));
    }

    #[test]
    fn search_dispatches_to_mismatch() {
        let table = TruffleTable::new(b"a");
        let vtable = mock_vtable();
        assert_eq!(vtable.search::<true>(&table, b"abcd"), Some(1));
    }

    #[test]
    fn search_translates_the_end_sentinel_to_none() {
        let table = TruffleTable::new(b"a");
        let vtable = mock_vtable();
        for haystack in [b"!".as_slice(), b""] {
            assert_eq!(vtable.search::<false>(&table, haystack), None);
            assert_eq!(vtable.search::<true>(&table, haystack), None);
        }
    }

    #[test]
    fn rsearch_dispatches_to_rfind() {
        let table = TruffleTable::new(b"a");
        let vtable = mock_vtable();
        assert_eq!(vtable.rsearch::<false>(&table, b"abcd"), Some(2));
    }

    #[test]
    fn rsearch_dispatches_to_rmismatch() {
        let table = TruffleTable::new(b"a");
        let vtable = mock_vtable();
        assert_eq!(vtable.rsearch::<true>(&table, b"abcd"), Some(3));
    }

    #[test]
    fn rsearch_translates_the_end_sentinel_to_none() {
        let table = TruffleTable::new(b"a");
        let vtable = mock_vtable();
        for haystack in [b"!".as_slice(), b""] {
            assert_eq!(vtable.rsearch::<false>(&table, haystack), None);
            assert_eq!(vtable.rsearch::<true>(&table, haystack), None);
        }
    }

    fn mock_vtable() -> VTable {
        VTable {
            find: entry::<0>,
            mismatch: entry::<1>,
            rfind: entry::<2>,
            rmismatch: entry::<3>,
        }
    }

    fn entry<const INDEX: usize>(low: &[u8; 16], high: &[u8; 16], haystack: &[u8]) -> usize {
        let table = TruffleTable::new(b"a");
        assert_eq!(low, table.lower());
        assert_eq!(high, table.upper());
        match haystack {
            b"abcd" => INDEX,
            b"!" | b"" => haystack.len(),
            _ => panic!("unexpected haystack: {haystack:?}"),
        }
    }
}
