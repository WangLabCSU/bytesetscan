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
        // Each byte is split as [s hhh llll]: s selects lower/upper,
        // hhh selects a membership bit, and llll selects a table entry.
        // For example, A = 0b0100_0001 and a = 0b0110_0001 both use lower[1].
        // They set bits 4 and 6, so lower[1] contains 0b0101_0000.
        // Decoding uses those bit POSITIONS (4 and 6), not the flags value.
        //
        // Decode lower first, then upper. base_flag restores bit 7, putting
        // every lower-half byte (0..=127) before every upper-half byte (128..=255).
        [(0b0000_0000, &self.lower), (0b1000_0000, &self.upper)]
            .into_iter()
            .flat_map(|(base_flag, table)| {
                // hhh has three bits: 1 << 3 is 8, so bit_index visits 0..=7.
                // Visit each bit position across all 16 entries before moving
                // to the next bit position. This yields 0..15 before 16..31,
                // and so on; visiting all bits of one entry first would put
                // byte 16 (entry 0 / bit 1) before byte 15 (entry 15 / bit 0).
                (0..1 << 3).flat_map(move |bit_index| {
                    table
                        .iter()
                        .enumerate()
                        .filter_map(move |(low_nibble, &bit_flags)| {
                            // Test whether this entry contains the current bit.
                            // A clear bit means this particular byte is absent;
                            // other set bits in the entry are visited separately.
                            if bit_flags & (1 << bit_index) != 0 {
                                // Put hhh in bits 4..6 and restore s in bit 7.
                                // The low four bits remain clear for the index.
                                let high_nibble = base_flag | (bit_index << 4);
                                // Restore llll from the table index. For lower[1]:
                                // bit 4 -> 0b0100_0000 | 1 = 0b0100_0001 (A),
                                // bit 6 -> 0b0110_0000 | 1 = 0b0110_0001 (a).
                                // Byte 0 is valid too: lower[0], bit 0 yields Some(0).
                                Some(high_nibble | low_nibble as u8)
                            } else {
                                None
                            }
                        })
                })
            })
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
        // Visit every table entry and flag. Construct its corresponding byte
        // and require exactly that flag to be set, with all other flags clear.
        for table_selector in 0..2 {
            for low_nibble in 0..16 {
                for bit_index in 0..8 {
                    // Lower/upper selects bit 7; the flag index supplies bits 4..6.
                    // The entry index supplies the low four bits.
                    let byte = ((table_selector << 7) | (bit_index << 4) | low_nibble) as u8;
                    let table = TruffleTable::new(&[byte]);
                    let mut expected = [[0; 16]; 2];
                    expected[table_selector][low_nibble] = 1 << bit_index;
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
            for (table_selector, mask) in [table.lower, table.upper].iter().enumerate() {
                for (low_nibble, &bit_flags) in mask.iter().enumerate() {
                    for bit_index in 0..8 {
                        // Lower/upper selects bit 7; the flag index supplies bits 4..6.
                        // The entry index supplies the low four bits.
                        let byte = ((table_selector << 7) | (bit_index << 4) | low_nibble) as u8;
                        assert_eq!(
                            bit_flags & (1 << bit_index) != 0,
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
            for (table_selector, mask) in [table.lower, table.upper].iter().enumerate() {
                for (low_nibble, &bit_flags) in mask.iter().enumerate() {
                    for bit_index in 0..8 {
                        // Lower/upper selects bit 7; the flag index supplies bits 4..6.
                        // The entry index supplies the low four bits.
                        let byte = (table_selector << 7) | (bit_index << 4) | low_nibble;
                        assert_eq!(
                            bit_flags & (1 << bit_index) != 0,
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
            for (table_selector, mask) in [table.lower, table.upper].iter().enumerate() {
                for (low_nibble, &bit_flags) in mask.iter().enumerate() {
                    for bit_index in 0..8 {
                        // Lower/upper selects bit 7; the flag index supplies bits 4..6.
                        // The entry index supplies the low four bits.
                        let byte = ((table_selector << 7) | (bit_index << 4) | low_nibble) as u8;
                        assert_eq!(
                            bit_flags & (1 << bit_index) != 0,
                            byte == member,
                            "byte={byte}"
                        );
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
            for (table_selector, mask) in [table.lower, table.upper].iter().enumerate() {
                for (low_nibble, &bit_flags) in mask.iter().enumerate() {
                    for bit_index in 0..8 {
                        // Lower/upper selects bit 7; the flag index supplies bits 4..6.
                        // The entry index supplies the low four bits.
                        let byte = ((table_selector << 7) | (bit_index << 4) | low_nibble) as u8;
                        assert_eq!(
                            bit_flags & (1 << bit_index) != 0,
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
        // Inspect every stored membership flag independently of needle().
        for (table_selector, mask) in [table.lower, table.upper].iter().enumerate() {
            for (low_nibble, &bit_flags) in mask.iter().enumerate() {
                for bit_index in 0..8 {
                    // Lower/upper selects bit 7; the flag index supplies bits 4..6.
                    // The entry index supplies the low four bits.
                    let byte = ((table_selector << 7) | (bit_index << 4) | low_nibble) as u8;
                    assert_eq!(
                        bit_flags & (1 << bit_index) != 0,
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
        for (table_selector, mask) in [table.lower, table.upper].iter().enumerate() {
            for (low_nibble, &bit_flags) in mask.iter().enumerate() {
                for bit_index in 0..8 {
                    // Lower/upper selects bit 7; the flag index supplies bits 4..6.
                    // The entry index supplies the low four bits.
                    let byte = ((table_selector << 7) | (bit_index << 4) | low_nibble) as u8;
                    assert_eq!(
                        bit_flags & (1 << bit_index) != 0,
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
        for (table_selector, mask) in [table.lower, table.upper].iter().enumerate() {
            for (low_nibble, &bit_flags) in mask.iter().enumerate() {
                for bit_index in 0..8 {
                    // Lower/upper selects bit 7; the flag index supplies bits 4..6.
                    // The entry index supplies the low four bits.
                    let byte = ((table_selector << 7) | (bit_index << 4) | low_nibble) as u8;
                    assert_eq!(
                        bit_flags & (1 << bit_index) != 0,
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
        // Inspect every stored membership flag independently of needle().
        for (table_selector, mask) in [table.lower, table.upper].iter().enumerate() {
            for (low_nibble, &bit_flags) in mask.iter().enumerate() {
                for bit_index in 0..8 {
                    // Lower/upper selects bit 7; the flag index supplies bits 4..6.
                    // The entry index supplies the low four bits.
                    let byte = ((table_selector << 7) | (bit_index << 4) | low_nibble) as u8;
                    assert_eq!(
                        bit_flags & (1 << bit_index) != 0,
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
                .filter(|byte| {
                    let high_nibble = byte >> 4;
                    let low_nibble = byte & 0b0000_1111;
                    (high_nibble == low_nibble) == on_diagonal
                })
                .collect();
            let table = TruffleTable::new(&needle);
            for (table_selector, mask) in [table.lower, table.upper].iter().enumerate() {
                for (low_nibble, &bit_flags) in mask.iter().enumerate() {
                    for bit_index in 0..8 {
                        // Lower/upper selects bit 7; the flag index supplies bits 4..6.
                        // The entry index supplies the low four bits.
                        let byte = ((table_selector << 7) | (bit_index << 4) | low_nibble) as u8;
                        assert_eq!(
                            bit_flags & (1 << bit_index) != 0,
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
                .filter(|byte| {
                    let high_nibble = byte >> 4;
                    let low_nibble = byte & 0b0000_1111;
                    (high_nibble == low_nibble) == on_diagonal
                })
                .collect();
            let table = TruffleTable::with_case(false, &needle);
            for (table_selector, mask) in [table.lower, table.upper].iter().enumerate() {
                for (low_nibble, &bit_flags) in mask.iter().enumerate() {
                    for bit_index in 0..8 {
                        // Lower/upper selects bit 7; the flag index supplies bits 4..6.
                        // The entry index supplies the low four bits.
                        let byte = ((table_selector << 7) | (bit_index << 4) | low_nibble) as u8;
                        assert_eq!(
                            bit_flags & (1 << bit_index) != 0,
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
                .filter(|byte| {
                    let high_nibble = byte >> 4;
                    let low_nibble = byte & 0b0000_1111;
                    (high_nibble == low_nibble) == on_diagonal
                })
                .collect();
            let table = TruffleTable::with_case(true, &needle);
            let expected: Vec<u8> = needle
                .iter()
                .flat_map(|byte| [byte.to_ascii_lowercase(), byte.to_ascii_uppercase()])
                .collect();
            for (table_selector, mask) in [table.lower, table.upper].iter().enumerate() {
                for (low_nibble, &bit_flags) in mask.iter().enumerate() {
                    for bit_index in 0..8 {
                        // Lower/upper selects bit 7; the flag index supplies bits 4..6.
                        // The entry index supplies the low four bits.
                        let byte = ((table_selector << 7) | (bit_index << 4) | low_nibble) as u8;
                        assert_eq!(
                            bit_flags & (1 << bit_index) != 0,
                            expected.contains(&byte),
                            "byte={byte}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn needle_decodes_each_membership_bit() {
        for expected_byte in 0..=u8::MAX {
            let mut table = TruffleTable {
                lower: [0; 16],
                upper: [0; 16],
            };

            // Read the three fields of [s hhh llll] separately:
            // s chooses the table, hhh chooses its bit, llll chooses its entry.
            let uses_upper_table = expected_byte & 0b1000_0000 != 0;
            let bit_index = (expected_byte >> 4) & 0b0000_0111;
            let low_nibble = expected_byte & 0b0000_1111;
            let mask = if uses_upper_table {
                &mut table.upper
            } else {
                &mut table.lower
            };
            // Only this byte is present. A wrong mapping must yield either
            // another byte or no byte, rather than being hidden by a full table.
            mask[low_nibble as usize] = 1 << bit_index;

            assert!(table.needle().eq([expected_byte]), "byte={expected_byte}");
        }
    }

    #[test]
    fn needle_reads_the_lower_half() {
        let mut lower = [0; 16];
        lower[0] = 1 << 0; // Byte 0: first byte of the lower half.
        lower[15] = 1 << 7; // Byte 127: last byte of the lower half.
        let table = TruffleTable {
            lower,
            upper: [0; 16],
        };
        assert!(table.needle().eq([0, 127]));
    }

    #[test]
    fn needle_reads_the_upper_half() {
        let mut upper = [0; 16];
        upper[0] = 1 << 0; // Byte 128: first byte of the upper half.
        upper[15] = 1 << 7; // Byte 255: last byte of the upper half.
        let table = TruffleTable {
            lower: [0; 16],
            upper,
        };
        assert!(table.needle().eq([128, 255]));
    }

    #[test]
    fn needle_is_empty_when_both_masks_are_clear() {
        let table = TruffleTable {
            lower: [0; 16],
            upper: [0; 16],
        };

        assert_eq!(table.needle().next(), None);
    }

    #[test]
    fn needle_yields_every_byte_when_both_masks_are_full() {
        let table = TruffleTable {
            lower: [0b1111_1111; 16],
            upper: [0b1111_1111; 16],
        };

        assert!(table.needle().eq(0..=255));
    }

    #[test]
    fn needle_yields_bytes_in_ascending_order() {
        let mut lower = [0; 16];
        // Entry order is deliberately opposite to byte order.
        lower[0b0000] = 1 << 1; // 0b0001_0000 = 16
        lower[0b1111] = 1 << 0; // 0b0000_1111 = 15
        let table = TruffleTable {
            lower,
            upper: [0; 16],
        };

        assert!(table.needle().eq([15, 16]));
    }

    #[test]
    fn needle_yields_each_set_bit_in_one_entry() {
        let mut lower = [0; 16];
        // A = 0b0100_0001 and a = 0b0110_0001 share entry 1, using bits 4 and 6.
        lower[1] = (1 << 4) | (1 << 6);
        let table = TruffleTable {
            lower,
            upper: [0; 16],
        };

        assert!(table.needle().eq(*b"Aa"));
    }
}
