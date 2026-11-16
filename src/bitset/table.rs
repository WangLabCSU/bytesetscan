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
        // A byte is [entry index: 2 bits][bit index: 6 bits]. For example,
        // A = 0b01_000001 uses entry 1, bit 1. The set bit's position supplies
        // the low six bits of the byte; the stored flag value does not.
        // Visit entries first, then their bit positions, so every byte from
        // an earlier entry precedes every byte from a later entry.
        self.bits
            .iter()
            .enumerate()
            .flat_map(|(entry_index, &bit_flags)| {
                // Each u64 has one membership flag for each of its 64 positions.
                (0..u64::BITS).filter_map(move |bit_index| {
                    if bit_flags & (1 << bit_index) != 0 {
                        // Place the entry index in bits 6..7 and the bit index
                        // in bits 0..5. Entry 0 / bit 0 yields byte 0 normally.
                        Some(((entry_index << 6) | bit_index as usize) as u8)
                    } else {
                        None
                    }
                })
            })
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::vec::Vec;

    use super::BitsetTable;

    #[test]
    fn new_ignores_duplicate_bytes() {
        for member in 0..=255u8 {
            let table = BitsetTable::new(&[member, member]);
            // Repeating a byte must set only its own membership flag.
            for (entry_index, &bits) in table.bits.iter().enumerate() {
                for bit_index in 0..64 {
                    // The entry index supplies the top two bits; the bit index
                    // within that u64 supplies the low six bits.
                    let byte = ((entry_index << 6) | bit_index) as u8;
                    assert_eq!(bits & (1 << bit_index) != 0, byte == member, "byte={byte}");
                }
            }
        }
    }

    #[test]
    fn new_preserves_distinct_members_in_repeated_input() {
        for (needle, expected) in [
            (b"baba".as_slice(), b"ab".as_slice()),
            // Repeated, unordered bytes from both halves of the byte range.
            (&[255, 0, 128, 255, 17], &[0, 17, 128, 255]),
        ] {
            let table = BitsetTable::new(needle);
            for (entry_index, &bits) in table.bits.iter().enumerate() {
                for bit_index in 0..64 {
                    // The entry index supplies the top two bits; the bit index
                    // within that u64 supplies the low six bits.
                    let byte = ((entry_index << 6) | bit_index) as u8;
                    assert_eq!(
                        bits & (1 << bit_index) != 0,
                        expected.contains(&byte),
                        "byte={byte}"
                    );
                }
            }
        }
    }

    #[test]
    fn new_leaves_all_flags_clear_for_an_empty_needle() {
        let table = BitsetTable::new(b"");
        assert_eq!(table.bits, [0; 4]);
    }

    #[test]
    fn new_encodes_the_full_byte_set() {
        let needle: Vec<u8> = (0..=255).collect();
        let table = BitsetTable::new(&needle);
        assert_eq!(table.bits, [u64::MAX; 4]);
    }

    #[test]
    fn new_encodes_exact_membership() {
        let needle = b"aB";
        let table = BitsetTable::new(needle);
        // Inspect every stored membership bit independently of needle().
        for (entry_index, &bits) in table.bits.iter().enumerate() {
            for bit_index in 0..64 {
                // The entry index supplies the top two bits; the bit index
                // within that u64 supplies the low six bits.
                let byte = ((entry_index << 6) | bit_index) as u8;
                assert_eq!(
                    bits & (1 << bit_index) != 0,
                    needle.contains(&byte),
                    "byte={byte}"
                );
            }
        }
    }

    #[test]
    fn with_case_false_keeps_ascii_cases_distinct() {
        // NUL and high bytes stay unchanged; A and z exercise both ASCII cases.
        let needle = [0, b'A', b'z', 128, 255];
        let table = BitsetTable::with_case(false, &needle);
        let expected = [0, b'A', b'z', 128, 255];
        for (entry_index, &bits) in table.bits.iter().enumerate() {
            for bit_index in 0..64 {
                // The entry index supplies the top two bits; the bit index
                // within that u64 supplies the low six bits.
                let byte = ((entry_index << 6) | bit_index) as u8;
                assert_eq!(
                    bits & (1 << bit_index) != 0,
                    expected.contains(&byte),
                    "byte={byte}"
                );
            }
        }
    }

    #[test]
    fn with_case_true_expands_letters_among_non_ascii_bytes() {
        // NUL and high bytes stay unchanged; A and z exercise both ASCII cases.
        let needle = [0, b'A', b'z', 128, 255];
        let table = BitsetTable::with_case(true, &needle);
        let expected = [0, b'A', b'Z', b'a', b'z', 128, 255];
        for (entry_index, &bits) in table.bits.iter().enumerate() {
            for bit_index in 0..64 {
                // The entry index supplies the top two bits; the bit index
                // within that u64 supplies the low six bits.
                let byte = ((entry_index << 6) | bit_index) as u8;
                assert_eq!(
                    bits & (1 << bit_index) != 0,
                    expected.contains(&byte),
                    "byte={byte}"
                );
            }
        }
    }

    #[test]
    fn with_case_true_expands_ascii_letters_only() {
        // @, [, ` and { sit directly outside the ASCII letter ranges.
        let needle = b"@Az[`{";
        let expected = b"@AZ[`az{";
        let table = BitsetTable::with_case(true, needle);
        // Inspect every stored membership bit independently of needle().
        for (entry_index, &bits) in table.bits.iter().enumerate() {
            for bit_index in 0..64 {
                // The entry index supplies the top two bits; the bit index
                // within that u64 supplies the low six bits.
                let byte = ((entry_index << 6) | bit_index) as u8;
                assert_eq!(
                    bits & (1 << bit_index) != 0,
                    expected.contains(&byte),
                    "byte={byte}"
                );
            }
        }
    }

    #[test]
    fn new_encodes_sets_with_sixteen_distinct_rows() {
        // Equal nibbles form a 16-byte diagonal; the other 240 bytes form
        // its complement. Both sets have sixteen distinct row patterns.
        for on_diagonal in [true, false] {
            let needle: Vec<u8> = (0..=255u8)
                .filter(|byte| {
                    let high_nibble = byte >> 4;
                    let low_nibble = byte & 0b0000_1111;
                    (high_nibble == low_nibble) == on_diagonal
                })
                .collect();
            let table = BitsetTable::new(&needle);
            for (entry_index, &bits) in table.bits.iter().enumerate() {
                for bit_index in 0..64 {
                    // The entry index supplies the top two bits; the bit index
                    // within that u64 supplies the low six bits.
                    let byte = ((entry_index << 6) | bit_index) as u8;
                    assert_eq!(
                        bits & (1 << bit_index) != 0,
                        needle.contains(&byte),
                        "byte={byte}"
                    );
                }
            }
        }
    }

    #[test]
    fn with_case_false_preserves_sets_with_sixteen_distinct_rows() {
        // Equal nibbles form a 16-byte diagonal; the other 240 bytes form
        // its complement. Both sets have sixteen distinct row patterns.
        for on_diagonal in [true, false] {
            let needle: Vec<u8> = (0..=255u8)
                .filter(|byte| {
                    let high_nibble = byte >> 4;
                    let low_nibble = byte & 0b0000_1111;
                    (high_nibble == low_nibble) == on_diagonal
                })
                .collect();
            let table = BitsetTable::with_case(false, &needle);
            for (entry_index, &bits) in table.bits.iter().enumerate() {
                for bit_index in 0..64 {
                    // The entry index supplies the top two bits; the bit index
                    // within that u64 supplies the low six bits.
                    let byte = ((entry_index << 6) | bit_index) as u8;
                    assert_eq!(
                        bits & (1 << bit_index) != 0,
                        needle.contains(&byte),
                        "byte={byte}"
                    );
                }
            }
        }
    }

    #[test]
    fn with_case_true_expands_sets_with_sixteen_distinct_rows() {
        // Equal nibbles form a 16-byte diagonal; the other 240 bytes form
        // its complement. Both sets have sixteen distinct row patterns.
        for on_diagonal in [true, false] {
            let needle: Vec<u8> = (0..=255u8)
                .filter(|byte| {
                    let high_nibble = byte >> 4;
                    let low_nibble = byte & 0b0000_1111;
                    (high_nibble == low_nibble) == on_diagonal
                })
                .collect();
            let table = BitsetTable::with_case(true, &needle);
            let expected: Vec<u8> = needle
                .iter()
                .flat_map(|byte| [byte.to_ascii_lowercase(), byte.to_ascii_uppercase()])
                .collect();
            for (entry_index, &bits) in table.bits.iter().enumerate() {
                for bit_index in 0..64 {
                    // The entry index supplies the top two bits; the bit index
                    // within that u64 supplies the low six bits.
                    let byte = ((entry_index << 6) | bit_index) as u8;
                    assert_eq!(
                        bits & (1 << bit_index) != 0,
                        expected.contains(&byte),
                        "byte={byte}"
                    );
                }
            }
        }
    }

    #[test]
    fn needle_decodes_each_membership_bit() {
        for expected_byte in 0..=u8::MAX {
            let mut table = BitsetTable { bits: [0; 4] };
            // Read the two fields separately: the high two bits choose the
            // entry, and the low six bits choose its membership flag.
            let entry_index = expected_byte >> 6;
            let bit_index = expected_byte & 0b0011_1111;
            // Only this byte is present; every other bit remains clear.
            table.bits[entry_index as usize] = 1 << bit_index;

            assert!(table.needle().eq([expected_byte]), "byte={expected_byte}");
        }
    }

    #[test]
    fn needle_reads_the_first_and_last_bit_of_each_entry() {
        // Each u64 stores membership for one consecutive range of 64 bytes.
        for (entry, first_byte, last_byte) in [
            (0, 0, 63),    // bits[0]: 0..=63
            (1, 64, 127),  // bits[1]: 64..=127
            (2, 128, 191), // bits[2]: 128..=191
            (3, 192, 255), // bits[3]: 192..=255
        ] {
            // Only this entry has bits set; the other entries yield no bytes.
            let mut bits = [0; 4];
            bits[entry] = (1 << 0) | (1 << 63);
            let table = BitsetTable { bits };

            assert!(table.needle().eq([first_byte, last_byte]), "entry={entry}");
        }
    }

    #[test]
    fn needle_is_empty_when_all_entries_are_clear() {
        let table = BitsetTable { bits: [0; 4] };

        assert_eq!(table.needle().next(), None);
    }

    #[test]
    fn needle_yields_every_byte_when_all_entries_are_full() {
        let table = BitsetTable {
            bits: [u64::MAX; 4],
        };

        assert!(table.needle().eq(0..=u8::MAX));
    }

    #[test]
    fn needle_yields_bytes_in_ascending_order() {
        let mut bits = [0; 4];
        // The last bit of entry 0 must precede the first bit of entry 1,
        // even though their bit indices alone have the opposite order.
        bits[0] = 1 << 63; // 0b00_111111 = 63
        bits[1] = 1 << 0; // 0b01_000000 = 64
        let table = BitsetTable { bits };

        assert!(table.needle().eq([63, 64]));
    }

    #[test]
    fn needle_yields_each_set_bit_in_one_entry() {
        let mut bits = [0; 4];
        // A = 0b01_000001 and a = 0b01_100001 both use entry 1.
        // Their low six bits select bit positions 1 and 33, respectively.
        bits[1] = (1 << 1) | (1 << 33);
        let table = BitsetTable { bits };

        assert!(table.needle().eq(*b"Aa"));
    }
}
