use clap::ValueEnum;

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(super) enum Hit {
    Miss,
    First,
    Middle,
    Last,
}

pub(super) struct Input {
    storage: Vec<u8>,
    start: usize,
    len: usize,
    expected: Option<usize>,
}

impl Input {
    pub(super) fn new(len: usize, hit: Hit, offset: usize) -> Self {
        // 255 is absent from every benchmark needle. Zero is present in all.
        // Reserve enough prefix space to reach the next 64-byte boundary and
        // then apply offset. Vec's allocation need not itself be aligned to 64.
        // At least one prefix byte remains for Hyperscan's reverse sentinel.
        let mut storage = vec![255; 64 + offset + len];
        let start = 64 - (storage.as_ptr() as usize % 64) + offset;
        let expected = if len == 0 {
            None
        } else {
            match hit {
                Hit::Miss => None,
                Hit::First => Some(0),
                Hit::Middle => Some(len / 2),
                Hit::Last => Some(len - 1),
            }
        };
        if let Some(position) = expected {
            storage[start + position] = 0;
        }
        Self {
            storage,
            start,
            len,
            expected,
        }
    }

    pub(super) fn haystack(&self) -> &[u8] {
        &self.storage[self.start..self.start + self.len]
    }

    pub(super) fn expected(&self) -> Option<usize> {
        self.expected
    }
}

#[cfg(test)]
mod tests {
    use super::{Hit, Input};

    #[test]
    fn miss_contains_no_member() {
        let input = Input::new(3, Hit::Miss, 0);
        assert_eq!(input.haystack(), [255, 255, 255]);
        assert_eq!(input.expected(), None);
    }

    #[test]
    fn first_places_the_member_at_the_start() {
        let input = Input::new(3, Hit::First, 0);
        assert_eq!(input.haystack(), [0, 255, 255]);
        assert_eq!(input.expected(), Some(0));
    }

    #[test]
    fn middle_places_the_member_between_nonmembers() {
        let input = Input::new(3, Hit::Middle, 0);
        assert_eq!(input.haystack(), [255, 0, 255]);
        assert_eq!(input.expected(), Some(1));
    }

    #[test]
    fn last_places_the_member_at_the_end() {
        let input = Input::new(3, Hit::Last, 0);
        assert_eq!(input.haystack(), [255, 255, 0]);
        assert_eq!(input.expected(), Some(2));
    }

    #[test]
    fn empty_input_has_no_match() {
        let input = Input::new(0, Hit::Last, 0);
        assert_eq!(input.haystack(), []);
        assert_eq!(input.expected(), None);
    }

    #[test]
    fn offset_sets_the_haystack_address_modulo_64() {
        for offset in 0..64 {
            let input = Input::new(3, Hit::Miss, offset);
            assert_eq!(input.haystack().as_ptr() as usize % 64, offset);
        }
    }

    #[test]
    fn haystack_keeps_a_prefix_for_reverse_search() {
        for offset in 0..64 {
            let input = Input::new(3, Hit::Miss, offset);
            assert!(input.haystack().as_ptr() as usize > input.storage.as_ptr() as usize);
        }
    }
}
