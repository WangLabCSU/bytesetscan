// Shared matching data for Truffle and TruffleEngine. Each mask records
// one half of the byte range: lower covers 0..=127, upper covers 128..=255.
#[derive(Debug, Clone)]
pub(super) struct TruffleTable {
    lower: [u8; 16],
    upper: [u8; 16],
}

impl TruffleTable {
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

    fn build(needle: &[u8], ignore_ascii_case: bool) -> Self {
        // Split each byte as [s hhh llll]: s selects lower (0) or upper (1),
        // llll selects one of 16 entries, and hhh selects one of its 8 bits.
        // Each of the 256 byte values therefore has its own membership bit.
        //
        // Entries store membership flags, not character codes. Even lower
        // can contain 0b1000_0000: 'p' = 0b0111_0000 sets bit 7 of lower[0],
        // although the character's own top bit is zero.
        let mut lower = [0; 16];
        let mut upper = [0; 16];
        let mut i = 0;
        while i < needle.len() {
            let byte = needle[i];
            let index = (byte & 0b0000_1111) as usize;
            // Shifting also moves the table selector (original bit 7) into
            // bit 3. Mask it off so only original bits 4..6 select the bit.
            let bit = 1 << ((byte >> 4) & 0b0000_0111);
            if (byte & 0b1000_0000) == 0 {
                lower[index] |= bit;
            } else {
                upper[index] |= bit;
            }
            if ignore_ascii_case && byte.is_ascii_alphabetic() {
                // Insert the opposite case so searches need no case conversion.
                // The alphabetic check prevents this XOR from changing nonletters.
                let other_case = byte ^ (b'a' ^ b'A');
                // The XOR flips only bit 5 (0b0010_0000), leaving the low four
                // bits, and therefore index, unchanged:
                //   'A' = 0b0100_0001 -> lower[1], bit 4
                //   'a' = 0b0110_0001 -> lower[1], bit 6
                // Both are ASCII letters, so their top bit is zero and both
                // use lower[index]. Unlike an arbitrary byte, other_case >> 4
                // is already in 4..=7: no & 0b0000_0111 is needed.
                // For 'A' -> 'a', add bit 6 while preserving bit 4.
                lower[index] |= 1 << (other_case >> 4);
            }
            i += 1;
        }
        Self { lower, upper }
    }

    /// Returns the masks for byte values `0..=127`.
    ///
    /// Byte `b` in this range belongs to the needle when
    /// `lower[(b & 0b00001111) as usize] & (1 << (b >> 4)) != 0`.
    pub(super) fn lower(&self) -> &[u8; 16] {
        &self.lower
    }

    /// Returns the masks for byte values `128..=255`.
    ///
    /// Byte `b` in this range belongs to the needle when
    /// `upper[(b & 0b00001111) as usize] & (1 << ((b >> 4) & 0b00000111)) != 0`.
    pub(super) fn upper(&self) -> &[u8; 16] {
        &self.upper
    }

    /// Returns an iterator over the needle in ascending byte order.
    ///
    /// This is the matching byte set, not the original constructor input.
    /// Each byte is yielded once. With ASCII case-insensitive matching, both
    /// cases of each letter are included.
    pub(super) fn needle(&self) -> impl Iterator<Item = u8> + '_ {
        (0..=u8::MAX).filter(|&byte| self.search(byte))
    }

    #[inline(always)]
    fn search(&self, byte: u8) -> bool {
        let index = (byte & 0b00001111) as usize;
        let flags = if byte < 128 {
            self.lower[index]
        } else {
            self.upper[index]
        };
        flags & (1 << ((byte >> 4) & 0b00000111)) != 0
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::vec::Vec;

    use super::TruffleTable;

    #[test]
    fn new_sets_all_bits_for_the_full_set() {
        let needle: Vec<u8> = (0..=255).collect();
        let table = TruffleTable::new(&needle);
        assert_eq!(table.lower, [0b1111_1111; 16]);
        assert_eq!(table.upper, [0b1111_1111; 16]);
    }

    #[test]
    fn new_sets_only_the_single_bytes_bit() {
        // Each table coordinate represents byte half*128 + bit*16 + column.
        // Build from that byte and require exactly this one bit to be set.
        for half in 0..2 {
            for column in 0..16 {
                for bit in 0..8 {
                    let byte = (half * 128 + bit * 16 + column) as u8;
                    let table = TruffleTable::new(&[byte]);
                    let mut expected = [[0; 16]; 2];
                    expected[half][column] = 1 << bit;
                    assert_eq!([table.lower, table.upper], expected, "byte={byte}");
                }
            }
        }
    }

    #[test]
    fn new_encodes_growing_sets() {
        // Keep 255 while adding the other bytes: {0,255}, {0,1,255}, ...,
        // all bytes. Check every bit, including those not yet in the input.
        let mut needle = Vec::from([255]);
        for added in 0..=255 {
            if added != 255 {
                needle.push(added);
            }
            let table = TruffleTable::new(&needle);
            for (half, mask) in [table.lower, table.upper].iter().enumerate() {
                for (column, &flags) in mask.iter().enumerate() {
                    for bit in 0..8 {
                        let byte = (half * 128 + bit * 16 + column) as u8;
                        assert_eq!(
                            flags & (1 << bit) != 0,
                            byte <= added || byte == 255,
                            "added={added}, byte={byte}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn new_encodes_shrinking_sets() {
        // Remove 0, then 1, ..., then 255 from the full set. Each rebuilt
        // table must contain only the remaining suffix, ending with no bits set.
        let all: Vec<u8> = (0..=255).collect();
        for removed in 0..256 {
            let remaining = &all[removed + 1..];
            let table = TruffleTable::new(remaining);
            for (half, mask) in [table.lower, table.upper].iter().enumerate() {
                for (column, &flags) in mask.iter().enumerate() {
                    for bit in 0..8 {
                        let byte = half * 128 + bit * 16 + column;
                        assert_eq!(
                            flags & (1 << bit) != 0,
                            byte > removed,
                            "removed={removed}, byte={byte}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn new_ignores_duplicate_bytes() {
        for member in 0..=255u8 {
            let table = TruffleTable::new(&[member, member]);
            // Repeating a byte must set only its own membership flag.
            for (half, mask) in [table.lower, table.upper].iter().enumerate() {
                for (column, &flags) in mask.iter().enumerate() {
                    for bit in 0..8 {
                        let byte = (half * 128 + bit * 16 + column) as u8;
                        assert_eq!(flags & (1 << bit) != 0, byte == member, "byte={byte}");
                    }
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
            let table = TruffleTable::new(needle);
            for (half, mask) in [table.lower, table.upper].iter().enumerate() {
                for (column, &flags) in mask.iter().enumerate() {
                    for bit in 0..8 {
                        let byte = (half * 128 + bit * 16 + column) as u8;
                        assert_eq!(
                            flags & (1 << bit) != 0,
                            expected.contains(&byte),
                            "byte={byte}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn new_encodes_exact_membership() {
        let needle = b"aB";
        let table = TruffleTable::new(needle);
        // In each half, column c and bit r encode byte half*128 + r*16 + c.
        // Inspect those stored bits without calling search() or needle().
        for (half, mask) in [table.lower, table.upper].iter().enumerate() {
            for (column, &flags) in mask.iter().enumerate() {
                for bit in 0..8 {
                    let byte = (half * 128 + bit * 16 + column) as u8;
                    assert_eq!(
                        flags & (1 << bit) != 0,
                        needle.contains(&byte),
                        "byte={byte}"
                    );
                }
            }
        }
    }

    #[test]
    fn with_case_false_keeps_ascii_cases_distinct() {
        // NUL and high bytes stay unchanged; A and z exercise both ASCII cases.
        let needle = [0, b'A', b'z', 128, 255];
        let table = TruffleTable::with_case(false, &needle);
        let expected = [0, b'A', b'z', 128, 255];
        for (half, mask) in [table.lower, table.upper].iter().enumerate() {
            for (column, &flags) in mask.iter().enumerate() {
                for bit in 0..8 {
                    let byte = (half * 128 + bit * 16 + column) as u8;
                    assert_eq!(
                        flags & (1 << bit) != 0,
                        expected.contains(&byte),
                        "byte={byte}"
                    );
                }
            }
        }
    }

    #[test]
    fn with_case_true_expands_letters_among_non_ascii_bytes() {
        // NUL and high bytes stay unchanged; A and z exercise both ASCII cases.
        let needle = [0, b'A', b'z', 128, 255];
        let table = TruffleTable::with_case(true, &needle);
        let expected = [0, b'A', b'Z', b'a', b'z', 128, 255];
        for (half, mask) in [table.lower, table.upper].iter().enumerate() {
            for (column, &flags) in mask.iter().enumerate() {
                for bit in 0..8 {
                    let byte = (half * 128 + bit * 16 + column) as u8;
                    assert_eq!(
                        flags & (1 << bit) != 0,
                        expected.contains(&byte),
                        "byte={byte}"
                    );
                }
            }
        }
    }

    #[test]
    fn with_case_true_expands_ascii_letters_only() {
        // @, [, ` and { sit directly outside the ASCII letter ranges.
        let needle = b"@Az[`{";
        let expected = b"@AZ[`az{";
        let table = TruffleTable::with_case(true, needle);
        // In each half, column c and bit r encode byte half*128 + r*16 + c.
        // Inspect those stored bits without calling search() or needle().
        for (half, mask) in [table.lower, table.upper].iter().enumerate() {
            for (column, &flags) in mask.iter().enumerate() {
                for bit in 0..8 {
                    let byte = (half * 128 + bit * 16 + column) as u8;
                    assert_eq!(
                        flags & (1 << bit) != 0,
                        expected.contains(&byte),
                        "byte={byte}"
                    );
                }
            }
        }
    }

    #[test]
    fn new_encodes_sets_with_sixteen_distinct_rows() {
        // Equal nibbles form a 16-byte diagonal; the other 240 bytes form
        // its complement. Both sets have sixteen distinct row patterns.
        for on_diagonal in [true, false] {
            let needle: Vec<u8> = (0..=255u8)
                .filter(|byte| (byte / 16 == byte % 16) == on_diagonal)
                .collect();
            let table = TruffleTable::new(&needle);
            for (half, mask) in [table.lower, table.upper].iter().enumerate() {
                for (column, &flags) in mask.iter().enumerate() {
                    for bit in 0..8 {
                        let byte = (half * 128 + bit * 16 + column) as u8;
                        assert_eq!(
                            flags & (1 << bit) != 0,
                            needle.contains(&byte),
                            "byte={byte}"
                        );
                    }
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
                .filter(|byte| (byte / 16 == byte % 16) == on_diagonal)
                .collect();
            let table = TruffleTable::with_case(false, &needle);
            for (half, mask) in [table.lower, table.upper].iter().enumerate() {
                for (column, &flags) in mask.iter().enumerate() {
                    for bit in 0..8 {
                        let byte = (half * 128 + bit * 16 + column) as u8;
                        assert_eq!(
                            flags & (1 << bit) != 0,
                            needle.contains(&byte),
                            "byte={byte}"
                        );
                    }
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
                .filter(|byte| (byte / 16 == byte % 16) == on_diagonal)
                .collect();
            let table = TruffleTable::with_case(true, &needle);
            let expected: Vec<u8> = needle
                .iter()
                .flat_map(|byte| [byte.to_ascii_lowercase(), byte.to_ascii_uppercase()])
                .collect();
            for (half, mask) in [table.lower, table.upper].iter().enumerate() {
                for (column, &flags) in mask.iter().enumerate() {
                    for bit in 0..8 {
                        let byte = (half * 128 + bit * 16 + column) as u8;
                        assert_eq!(
                            flags & (1 << bit) != 0,
                            expected.contains(&byte),
                            "byte={byte}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn search_reads_the_lower_half() {
        let mut lower = [0; 16];
        lower[0] = 1 << 0; // Byte 0: first byte of the lower half.
        lower[15] = 1 << 7; // Byte 127: last byte of the lower half.
        let table = TruffleTable {
            lower,
            upper: [0; 16],
        };
        for byte in 0..=255 {
            assert_eq!(table.search(byte), byte == 0 || byte == 127, "byte={byte}");
        }
    }

    #[test]
    fn search_reads_the_upper_half() {
        let mut upper = [0; 16];
        upper[0] = 1 << 0; // Byte 128: first byte of the upper half.
        upper[15] = 1 << 7; // Byte 255: last byte of the upper half.
        let table = TruffleTable {
            lower: [0; 16],
            upper,
        };
        for byte in 0..=255 {
            assert_eq!(
                table.search(byte),
                byte == 128 || byte == 255,
                "byte={byte}"
            );
        }
    }

    #[test]
    fn needle_yields_bytes_in_ascending_order() {
        let mut lower = [0; 16];
        let mut upper = [0; 16];
        // Lower-half bytes: column 0/bit 0 = 0, column 1/bit 4 = A,
        // column 2/bit 6 = b. The upper half adds 128 to those coordinates.
        lower[0] = 1;
        lower[1] = 1 << 4;
        lower[2] = 1 << 6;
        upper[0] = 1; // 128
        upper[15] = 1 << 7; // 255
        let table = TruffleTable { lower, upper };

        assert!(table.needle().eq([0, b'A', b'b', 128, 255]));
    }
}
