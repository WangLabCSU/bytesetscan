// Shared matching data for Bitset and BitsetEngine. The four u64 entries cover
// byte values 0..=63, 64..=127, 128..=191, and 192..=255, respectively.
#[derive(Debug, Clone)]
pub(super) struct BitsetTable {
    bits: [u64; 4],
}

impl BitsetTable {
    /// Creates a table for the bytes in `needle`.
    ///
    /// An empty needle matches no bytes. Duplicate bytes are ignored.
    pub(super) fn new(needle: &[u8]) -> Self {
        Self::build(needle, false)
    }

    /// Creates a table that can ignore ASCII letter case.
    ///
    /// When `ignore_case` is `true`, ASCII letters match either case; other
    /// bytes match exactly. When it is `false`, this behaves like [`Self::new`].
    pub(super) fn with_case(ignore_case: bool, needle: &[u8]) -> Self {
        Self::build(needle, ignore_case)
    }

    fn build(needle: &[u8], ignore_case: bool) -> Self {
        let mut bits = [0u64; 4];
        let mut i = 0;
        while i < needle.len() {
            let byte = needle[i];
            // The high two bits select one of four entries. The low six bits
            // select one of its 64 bits, giving every byte its own position.
            bits[(byte >> 6) as usize] |= 1 << (byte & 0b0011_1111);
            if ignore_case && byte.is_ascii_alphabetic() {
                // ASCII letter cases differ only in bit 5. Set the opposite
                // case's membership bit too; nonletters remain unchanged.
                let other_case = byte ^ (b'a' ^ b'A');
                bits[(other_case >> 6) as usize] |= 1 << (other_case & 0b0011_1111);
            }
            i += 1;
        }
        Self { bits }
    }

    /// Returns the membership bits for all 256 byte values.
    pub(super) fn bits(&self) -> &[u64; 4] {
        &self.bits
    }

    /// Returns an iterator over the needle in ascending byte order.
    ///
    /// This is the matching byte set, not the original constructor input.
    /// Each byte is yielded once. With ASCII case-insensitive matching, both
    /// cases of each letter are included.
    pub(super) fn needle(&self) -> impl Iterator<Item = u8> + '_ {
        (0..=u8::MAX).filter(|&byte| self.contains(byte))
    }

    #[inline(always)]
    fn contains(&self, byte: u8) -> bool {
        // A byte always selects entry 0..=3 and bit 0..=63, including 0b1111_1111.
        self.bits[(byte >> 6) as usize] & (1 << (byte & 0b0011_1111)) != 0
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::vec::Vec;

    use super::BitsetTable;

    #[test]
    fn new_encodes_exact_membership() {
        for_each_needle(|needle| {
            let table = BitsetTable::new(needle);
            let actual: Vec<u8> = table.needle().collect();
            for byte in 0..=255u8 {
                assert_eq!(
                    actual.contains(&byte),
                    needle.contains(&byte),
                    "byte {byte}"
                );
            }
        });
    }

    #[test]
    fn with_case_false_preserves_exact_membership() {
        for_each_needle(|needle| {
            let table = BitsetTable::with_case(false, needle);
            let actual: Vec<u8> = table.needle().collect();
            for byte in 0..=255u8 {
                assert_eq!(
                    actual.contains(&byte),
                    needle.contains(&byte),
                    "byte {byte}"
                );
            }
        });
    }

    #[test]
    fn with_case_true_expands_ascii_letters_only() {
        for_each_needle(|needle| {
            let table = BitsetTable::with_case(true, needle);
            let actual: Vec<u8> = table.needle().collect();
            let expected: Vec<u8> = needle
                .iter()
                .flat_map(|byte| [byte.to_ascii_lowercase(), byte.to_ascii_uppercase()])
                .collect();
            for byte in 0..=255u8 {
                assert_eq!(
                    actual.contains(&byte),
                    expected.contains(&byte),
                    "byte {byte}"
                );
            }
        });
    }

    // Supply inputs to each constructor test; the caller checks its expected set.
    fn for_each_needle(mut visit: impl FnMut(&[u8])) {
        // Duplicate each possible byte to exercise repeated insertion at every
        // bit position, including all ASCII letters and nonletters.
        for member in 0..=255u8 {
            let needle = [member, member];
            visit(&needle);
        }

        // Every possible byte: all membership bits should be set.
        let all: Vec<u8> = (0..=255u8).collect();
        // n << 4 fills the high four bits; adding n fills the low four bits.
        // Since n < 16, the two parts do not overlap and addition has no carry.
        // For n = 3: (0b0011 << 4) + 0b0011 = 0b0011_0011.
        // These 16 bytes form a sparse set spanning all four entries.
        let diagonal: Vec<u8> = (0..16).map(|n| (n << 4) + n).collect();
        // The other 240 bytes form a dense set. Case expansion can add an
        // excluded letter back when its opposite case belongs to this input.
        let complement: Vec<u8> = all
            .iter()
            .copied()
            .filter(|b| !diagonal.contains(b))
            .collect();
        for needle in [
            b"".as_slice(),                        // Empty set.
            b"baba",                               // Repeated, unordered letters.
            &[0, 63, 64, 127, 128, 191, 192, 255], // Both ends of each entry.
            &diagonal,                             // Sparse set.
            &complement,                           // Dense set.
            &all,                                  // Full set.
        ] {
            visit(needle);
        }
    }

    #[test]
    fn needle_yields_bytes_in_ascending_order() {
        let table = BitsetTable::new(&[255, b'b', 0, b'A', 128]);
        assert!(table.needle().eq([0, b'A', b'b', 128, 255]));
    }

    #[test]
    fn needle_omits_duplicate_input_bytes() {
        let table = BitsetTable::new(b"aaaa");
        assert!(table.needle().eq(*b"a"));
    }

    #[test]
    fn contains_reads_bits_across_all_entries() {
        // Set each entry's first and last bit directly so the membership test
        // does not depend on construction producing the right layout.
        let table = BitsetTable {
            bits: [0b1 | (0b1 << 63); 4],
        };
        for byte in 0..=255 {
            assert_eq!(
                table.contains(byte),
                [0, 63, 64, 127, 128, 191, 192, 255].contains(&byte)
            );
        }
    }
}
