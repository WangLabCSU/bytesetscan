// Shared matching data for Shufti and ShuftiEngine. The two masks occupy
// 32 bytes and contain no backend state. Keeping construction here lets both
// search paths use the same representation while retaining separate dispatch.
#[derive(Debug, Clone)]
pub(super) struct ShuftiTable {
    low: [u8; 16],
    high: [u8; 16],
}

impl ShuftiTable {
    /// Builds a table for the byte set in `needle`.
    ///
    /// Empty needles are accepted and duplicate bytes are ignored.
    /// Returns `None` if the set cannot be represented. Any set of ASCII bytes
    /// fits; see [choosing a matcher](crate#choosing-a-matcher) for non-ASCII sets.
    /// Use [`crate::truffle::Truffle`] to accept arbitrary byte sets.
    pub(super) fn new(needle: &[u8]) -> Option<Self> {
        Self::with_case(false, needle)
    }

    /// Builds a table for `needle` with optional ASCII case folding.
    ///
    /// If `ignore_case` is `true`, ASCII letters match either case. Otherwise,
    /// all bytes are matched exactly.
    /// Nonletters and non-ASCII bytes are matched exactly. Empty needles are
    /// accepted and duplicate bytes are ignored.
    /// Returns `None` if the expanded set exceeds the limits in [`Self::new`].
    pub(super) fn with_case(ignore_case: bool, needle: &[u8]) -> Option<Self> {
        Self::build(needle, ignore_case)
    }

    fn build(needle: &[u8], ignore_ascii_case: bool) -> Option<Self> {
        // Group rows with identical low-nibble sets under a shared flag.
        //
        // Use needle = b"ABRabr" as a running example. The algorithm accepts
        // arbitrary u8 values. Split each allowed byte into two parts:
        //   b >> 4 equals b / 16 and gives the array index h into rows;
        //   b & 0b00001111 equals b % 16 and gives the low-four-bit value l.
        // Both are in 0..=15. Each of the 16 rows is a u16 whose 16 bits record
        // which values of l are allowed. Bits are numbered from zero: bit l
        // is set exactly when byte h * 16 + l is in needle.
        //
        // A has byte value 65: h = 4, l = 1, so set bit 1 of rows[4].
        // B has byte value 66: h = 4, l = 2, so also set bit 2 of rows[4].
        // Together they give rows[4] = 0b00000000_00000010 |
        // 0b00000000_00000100 = 0b00000000_00000110.
        // After processing b"ABRabr", the nonempty row masks are:
        //
        // h | allowed bytes | allowed l | rows[h]
        // --+---------------+-----------+---------------------
        // 4 | A, B          | 1, 2      | 0b00000000_00000110
        // 5 | R             | 2         | 0b00000000_00000100
        // 6 | a, b          | 1, 2      | 0b00000000_00000110
        // 7 | r             | 2         | 0b00000000_00000100
        let mut rows = [0u16; 16];

        let mut i = 0;
        while i < needle.len() {
            let b = needle[i];

            rows[(b >> 4) as usize] |= 1 << (b & 0b00001111);

            if ignore_ascii_case && b.is_ascii_alphabetic() {
                // ASCII letters differ only in this bit. Insert the other
                // case now so every matching operation can use the tables
                // directly. Nonletters must not acquire extra matches.
                let other_case = b ^ (b'a' ^ b'A');
                rows[(other_case >> 4) as usize] |= 1 << (other_case & 0b00001111);
            }

            i += 1;
        }

        let mut low = [0u8; 16];
        let mut high = [0u8; 16];

        // Rows with exactly the same allowed low values share one group.
        // patterns[group] stores that group's set of allowed low values.
        // used counts the groups already created; only patterns[0..used]
        // contains assigned groups.
        //
        // Visiting h in ascending order gives the following for our example:
        // h | pattern             | group | flag       | action
        // --+---------------------+-------+------------+-------
        // 4 | 0b00000000_00000110  | 0     | 0b00000001 | create
        // 5 | 0b00000000_00000100  | 1     | 0b00000010 | create
        // 6 | 0b00000000_00000110  | 0     | 0b00000001 | reuse
        // 7 | 0b00000000_00000100  | 1     | 0b00000010 | reuse
        //
        // rows[4] and rows[6] both allow low values 1 and 2, so these uppercase
        // and lowercase bytes share a group. Flag values follow group creation
        // order; a particular flag is not tied to a fixed ASCII range.
        let mut patterns = [0u16; 8];
        let mut used = 0usize;

        let mut h = 0;
        while h < 16 {
            let pattern = rows[h];

            // Empty rows need no group and consume no flag. Their high[h]
            // stays zero, rejecting every byte with this high-four-bit value.
            if pattern != 0 {
                // Search the sets already stored. On exit:
                //   group < used: an identical set exists; reuse its group;
                //   group == used: no set matches; create a new group.
                // Only identical sets may share a flag. Merging {1, 2} and
                // {2} in our example would also accept Q (h = 5, l = 1).
                let mut group = 0;

                while group < used && patterns[group] != pattern {
                    group += 1;
                }

                if group == used {
                    if used == 8 {
                        return None;
                    }
                    patterns[used] = pattern;
                    used += 1;
                }

                // Select the bit at this group's position:
                // group | flag
                // ------+-----------
                // 0     | 0b00000001
                // 1     | 0b00000010
                // 2     | 0b00000100
                // 3     | 0b00001000
                // 4     | 0b00010000
                // 5     | 0b00100000
                // 6     | 0b01000000
                // 7     | 0b10000000
                //
                // Bits in pattern identify allowed low values; the bit in
                // flag identifies a group. These are different meanings.
                // Equal patterns reuse the same group and therefore the same flag.
                let flag = 1 << group;

                // high[h] selects the group for this high-four-bit value.
                // Here high[4] = high[6] = 0b00000001;
                // high[5] = high[7] = 0b00000010.
                high[h] = flag;

                // low[l] records all groups that allow low value l. Add this
                // group's flag only where bit l is set in pattern.
                // Group 0 allows l = 1 and 2: OR 0b00000001 into low[1] and low[2].
                // Group 1 allows l = 2: OR 0b00000010 into low[2].
                // low[2] = 0b00000001 | 0b00000010 = 0b00000011 means both
                // allow low value 2. Adding an existing flag again changes nothing.
                let mut l = 0;
                while l < 16 {
                    // 1 << l selects bit l. For 0b00000000_00000110:
                    // l = 1: AND 0b00000000_00000010 is nonzero, so it is allowed;
                    // l = 0: AND 0b00000000_00000001 is zero, so it is rejected.
                    if pattern & (1 << l) != 0 {
                        low[l] |= flag;
                    }

                    l += 1;
                }
            }

            h += 1;
        }

        Some(Self { low, high })
    }

    /// Returns the table indexed by a byte's low four bits.
    ///
    /// Byte `b` belongs to the needle when
    /// `low[(b & 0b00001111) as usize] & high[(b >> 4) as usize] != 0`, where `low`
    /// and `high` are returned by this method and [`Self::high`].
    pub(super) fn low(&self) -> &[u8; 16] {
        &self.low
    }

    /// Returns the table indexed by a byte's high four bits.
    ///
    /// Combine its entries with those from [`Self::low`] to test membership.
    pub(super) fn high(&self) -> &[u8; 16] {
        &self.high
    }

    /// Returns an iterator over the needle in ascending byte order.
    ///
    /// This is the matching byte set, not the original constructor input.
    /// Each byte is yielded once. With ASCII case-insensitive matching, both
    /// cases of each letter are included.
    pub(super) fn needle(&self) -> impl Iterator<Item = u8> + '_ {
        // Table indices supply the byte's high and low four bits. The entries
        // contain group flags: their bit positions do not encode byte values.
        // For example, low[1], high[4] and high[6] sharing bit 7 represents
        // A = 0b0100_0001 and a = 0b0110_0001, regardless of that flag's position.
        //
        // Visit high nibbles first, then low nibbles, to preserve byte order:
        // all bytes with high nibble 0 precede all bytes with high nibble 1.
        self.high
            .iter()
            .enumerate()
            .flat_map(move |(high_nibble, &high_flags)| {
                self.low
                    .iter()
                    .enumerate()
                    .filter_map(move |(low_nibble, &low_flags)| {
                        // A shared group must allow BOTH nibble values. Merely
                        // having nonzero entries is insufficient if their flags
                        // name different groups.
                        if low_flags & high_flags != 0 {
                            // Restore the byte from the two indices. Each pair
                            // is visited once, including (0, 0) for byte zero.
                            Some(((high_nibble << 4) | low_nibble) as u8)
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

    use super::ShuftiTable;

    #[test]
    fn new_leaves_all_flags_clear_for_an_empty_needle() {
        let table = ShuftiTable::new(b"").unwrap();
        assert_eq!(table.low, [0; 16]);
        assert_eq!(table.high, [0; 16]);
    }

    #[test]
    fn new_encodes_the_full_byte_set() {
        let needle: Vec<u8> = (0..=255).collect();
        let table = ShuftiTable::new(&needle).unwrap();
        // All rows share the same group, whose flag is bit 0.
        assert_eq!(table.low, [1; 16]);
        assert_eq!(table.high, [1; 16]);
    }

    #[test]
    fn new_accepts_eight_distinct_groups() {
        // Each byte has equal high and low nibbles: 0x00, 0x11, ..., 0x77.
        // Each row allows a different low nibble, requiring all eight flags.
        let needle: Vec<u8> = (0..8).map(|n| (n << 4) + n).collect();
        assert!(ShuftiTable::new(&needle).is_some());
    }

    #[test]
    fn new_reuses_a_group_after_all_eight_flags_are_assigned() {
        let mut needle: Vec<u8> = (0..8).map(|n| (n << 4) + n).collect();
        // Row 8 allows only low nibble 0, so it shares row 0's group.
        needle.push(128);
        assert!(ShuftiTable::new(&needle).is_some());
    }

    #[test]
    fn new_rejects_nine_distinct_groups() {
        // n << 4 puts n in the high four bits; adding n fills the low four bits.
        // For n = 3: (0b0011 << 4) + 0b0011 = 0b0011_0011.
        // Since n < 16, these parts do not overlap, so the addition has no carry.
        //
        // Shufti rows share a group only when their allowed-low-value SETS match.
        // Here each row contains just its own low value:
        //
        // high row           | 0   | 1   | 2   | ...
        // allowed low values | {0} | {1} | {2} | ...
        //
        // The sets differ, so every row needs a separate group flag.
        // Nine distinct sets require nine flags, but only eight are available.
        let needle: Vec<u8> = (0..9).map(|n| (n << 4) + n).collect();
        assert!(ShuftiTable::new(&needle).is_none());
    }

    #[test]
    fn with_case_rejects_unrepresentable_needles() {
        // n << 4 puts n in the high four bits; adding n fills the low four bits.
        // For n = 3: (0b0011 << 4) + 0b0011 = 0b0011_0011.
        // These eight different sets need eight separate group flags.
        // Setting bit 7 puts those rows in 8..=15. Byte 8 adds row 0 with
        // a ninth distinct low-value set. None are ASCII letters, so enabling
        // case expansion leaves all nine groups unchanged.
        let needle: Vec<u8> = (0..8).map(|n| 128 | ((n << 4) + n)).chain([8]).collect();
        for ignore_case in [false, true] {
            assert!(ShuftiTable::with_case(ignore_case, &needle).is_none());
        }
    }

    #[test]
    fn new_encodes_exact_membership() {
        let needle = b"aB";
        let table = ShuftiTable::new(needle).unwrap();
        // Read every high/low coordinate directly. Group flag numbers
        // may differ between valid encodings; their intersections must
        // represent exactly the input set.
        for (high_nibble, &high_flags) in table.high.iter().enumerate() {
            for (low_nibble, &low_flags) in table.low.iter().enumerate() {
                // The table indices supply the byte's high and low four bits.
                let byte = ((high_nibble << 4) | low_nibble) as u8;
                assert_eq!(
                    low_flags & high_flags != 0,
                    needle.contains(&byte),
                    "byte={byte}"
                );
            }
        }
    }

    #[test]
    fn new_encodes_single_byte_masks() {
        // BuildMask1: 'a' has low nibble 1 and high nibble 6. Only those
        // entries may contain the single group flag; all others stay clear.
        let table = ShuftiTable::new(b"a").unwrap();
        assert_eq!(table.low, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(table.high, [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn new_excludes_crossed_nibble_pairs() {
        // BuildMask2: combining 'a's low nibble with 'B's high nibble
        // produces 'A'; the opposite combination produces 'b'. Neither
        // crossed pair may match, even though both input bytes must match.
        let table = ShuftiTable::new(b"aB").unwrap();
        assert_ne!(table.low[1] & table.high[6], 0); // a
        assert_ne!(table.low[2] & table.high[4], 0); // B
        assert_eq!(table.low[1] & table.high[4], 0); // A
        assert_eq!(table.low[2] & table.high[6], 0); // b
    }

    #[test]
    fn new_encodes_four_letter_masks() {
        // BuildMask4: both low nibbles are members in both high-nibble rows.
        let table = ShuftiTable::new(b"aBAb").unwrap();
        assert_ne!(table.low[1] & table.high[6], 0); // a
        assert_ne!(table.low[1] & table.high[4], 0); // A
        assert_ne!(table.low[2] & table.high[6], 0); // b
        assert_ne!(table.low[2] & table.high[4], 0); // B
    }

    #[test]
    fn new_ignores_duplicate_bytes() {
        for member in 0..=255u8 {
            let table = ShuftiTable::new(&[member, member]).unwrap();
            // Repeating a byte must set only its own membership flag.
            for (high_nibble, &high_flags) in table.high.iter().enumerate() {
                for (low_nibble, &low_flags) in table.low.iter().enumerate() {
                    // The table indices supply the byte's high and low four bits.
                    let byte = ((high_nibble << 4) | low_nibble) as u8;
                    assert_eq!(low_flags & high_flags != 0, byte == member, "byte={byte}");
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
            let table = ShuftiTable::new(needle).unwrap();
            for (high_nibble, &high_flags) in table.high.iter().enumerate() {
                for (low_nibble, &low_flags) in table.low.iter().enumerate() {
                    // The table indices supply the byte's high and low four bits.
                    let byte = ((high_nibble << 4) | low_nibble) as u8;
                    assert_eq!(
                        low_flags & high_flags != 0,
                        expected.contains(&byte),
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
        let table = ShuftiTable::with_case(false, &needle).unwrap();
        let expected = [0, b'A', b'z', 128, 255];
        for (high_nibble, &high_flags) in table.high.iter().enumerate() {
            for (low_nibble, &low_flags) in table.low.iter().enumerate() {
                // The table indices supply the byte's high and low four bits.
                let byte = ((high_nibble << 4) | low_nibble) as u8;
                assert_eq!(
                    low_flags & high_flags != 0,
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
        let table = ShuftiTable::with_case(true, &needle).unwrap();
        let expected = [0, b'A', b'Z', b'a', b'z', 128, 255];
        for (high_nibble, &high_flags) in table.high.iter().enumerate() {
            for (low_nibble, &low_flags) in table.low.iter().enumerate() {
                // The table indices supply the byte's high and low four bits.
                let byte = ((high_nibble << 4) | low_nibble) as u8;
                assert_eq!(
                    low_flags & high_flags != 0,
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
        let table = ShuftiTable::with_case(true, needle).unwrap();
        // Read every high/low coordinate directly. Group flag numbers
        // may differ between valid encodings; their intersections must
        // represent exactly the expected set.
        for (high_nibble, &high_flags) in table.high.iter().enumerate() {
            for (low_nibble, &low_flags) in table.low.iter().enumerate() {
                // The table indices supply the byte's high and low four bits.
                let byte = ((high_nibble << 4) | low_nibble) as u8;
                assert_eq!(
                    low_flags & high_flags != 0,
                    expected.contains(&byte),
                    "byte={byte}"
                );
            }
        }
    }

    #[test]
    fn needle_reconstructs_each_byte_from_its_nibbles() {
        for expected_byte in 0..=u8::MAX {
            let mut table = ShuftiTable {
                low: [0; 16],
                high: [0; 16],
            };
            let low_nibble = expected_byte & 0b0000_1111;
            let high_nibble = expected_byte >> 4;
            // Only these two indices share a group. Using flag bit 7 for
            // every byte checks that the indices determine the returned byte.
            table.low[low_nibble as usize] = 1 << 7;
            table.high[high_nibble as usize] = 1 << 7;

            assert!(table.needle().eq([expected_byte]), "byte={expected_byte}");
        }
    }

    #[test]
    fn needle_does_not_mix_distinct_groups() {
        let mut low = [0; 16];
        let mut high = [0; 16];
        low[1] = 1 << 0;
        high[4] = 1 << 0; // A = 0b0100_0001, in group 0.
        low[2] = 1 << 1;
        high[6] = 1 << 1; // b = 0b0110_0010, in group 1.
        let table = ShuftiTable { low, high };

        // The crossed pairs B (high 4, low 2) and a (high 6, low 1) have
        // nonzero entries but no common group, so neither may be returned.
        assert!(table.needle().eq(*b"Ab"));
    }

    #[test]
    fn needle_reuses_a_group_for_multiple_high_nibbles() {
        let mut low = [0; 16];
        let mut high = [0; 16];
        // A and a share low nibble 1, with high nibbles 4 and 6.
        low[1] = 1 << 7;
        high[4] = 1 << 7; // A
        high[6] = 1 << 7; // a
        let table = ShuftiTable { low, high };

        assert!(table.needle().eq(*b"Aa"));
    }

    #[test]
    fn needle_is_empty_when_both_tables_are_clear() {
        let table = ShuftiTable {
            low: [0; 16],
            high: [0; 16],
        };

        assert_eq!(table.needle().next(), None);
    }

    #[test]
    fn needle_yields_every_byte_when_all_entries_share_a_group() {
        // Every high/low pair shares group 0, so every byte belongs to the set.
        let table = ShuftiTable {
            low: [1 << 0; 16],
            high: [1 << 0; 16],
        };

        assert!(table.needle().eq(0..=u8::MAX));
    }

    #[test]
    fn needle_yields_bytes_in_ascending_order() {
        let mut low = [0; 16];
        let mut high = [0; 16];
        // Low-nibble order and group order both put 16 before 15. The iterator
        // must instead visit high nibble 0 before high nibble 1.
        low[0b1111] = 1 << 7;
        high[0b0000] = 1 << 7; // 0b0000_1111 = 15
        low[0b0000] = 1 << 0;
        high[0b0001] = 1 << 0; // 0b0001_0000 = 16
        let table = ShuftiTable { low, high };

        assert!(table.needle().eq([15, 16]));
    }
}
