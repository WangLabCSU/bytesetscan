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
        // Enumerate byte values directly rather than storing a byte-universe
        // array. Each pair of table lookups tests the reconstructed membership;
        // no search backend or original input is needed for this operation.
        (0..=u8::MAX).filter(|&byte| {
            self.low[(byte & 0b00001111) as usize] & self.high[(byte >> 4) as usize] != 0
        })
    }
}

#[cfg(test)]
mod tests {

    extern crate std;

    use std::vec::Vec;

    use super::ShuftiTable;

    #[test]
    fn new_accepts_eight_distinct_groups() {
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
        // Eight distinct sets use all eight available group flags.
        let needle: Vec<u8> = (0..8).map(|n| (n << 4) + n).collect();
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
    fn new_reuses_a_group_after_all_eight_flags_are_assigned() {
        let mut needle: Vec<u8> = (0..8).map(|n| (n << 4) + n).collect();
        // 128 = 0b1000_0000 adds row 8 with low value 0. Its pattern
        // equals row 0's {0}, so it reuses that flag instead of needing a ninth.
        needle.push(128);
        assert!(ShuftiTable::new(&needle).is_some());
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
    fn needle_yields_bytes_in_ascending_order() {
        let table = ShuftiTable::new(&[255, b'b', 0, b'A', 128]).unwrap();
        assert!(table.needle().eq([0, b'A', b'b', 128, 255]));
    }

    #[test]
    fn needle_omits_duplicate_input_bytes() {
        let table = ShuftiTable::new(b"aaaa").unwrap();
        assert!(table.needle().eq(*b"a"));
    }

    fn for_each_needle(mut visit: impl FnMut(&[u8])) {
        // Exercise duplicate handling for every possible byte value.
        for member in 0..=255u8 {
            let needle = [member, member];
            visit(&needle);
        }

        let all: Vec<u8> = (0..=255u8).collect();
        // n << 4 puts n in the high four bits; adding n fills the low four bits.
        // For n = 3: (0b0011 << 4) + 0b0011 = 0b0011_0011.
        // These eight different sets need eight separate group flags.
        let groups: Vec<u8> = (0..8).map(|n| (n << 4) + n).collect();
        // 128 = 0b1000_0000 adds row 8 with the same low-value set {0}
        // as row 0, so the extra row can reuse an existing group flag.
        let reused: Vec<u8> = groups.iter().copied().chain([128]).collect();
        for needle in [
            b"".as_slice(),
            b"baba",
            &[255, 0, 128, 255, 17],
            &[0, b'A', b'z', 128, 255],
            &groups,
            &reused,
            &all,
        ] {
            visit(needle);
        }
    }

    #[test]
    fn new_encodes_exact_membership() {
        for_each_needle(|needle| {
            let table = ShuftiTable::new(needle).unwrap();
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
            let table = ShuftiTable::with_case(false, needle).unwrap();
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
            let table = ShuftiTable::with_case(true, needle).unwrap();
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
}
