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
    fn new_encodes_exact_membership() {
        for_each_needle(|needle| {
            let table = TruffleTable::new(needle);
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
            let table = TruffleTable::with_case(false, needle);
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
            let table = TruffleTable::with_case(true, needle);
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
        // Viewing high/low nibbles as row/column coordinates, these 16 bytes
        // lie on the diagonal. They span both table halves and all 16 indices.
        let diagonal: Vec<u8> = (0..16).map(|n| (n << 4) + n).collect();
        // The remaining 240 bytes give a dense input opposite to the sparse
        // diagonal. With case folding, an excluded letter can be added back
        // if its opposite case belongs to this input.
        let complement: Vec<u8> = all
            .iter()
            .copied()
            .filter(|b| !diagonal.contains(b))
            .collect();
        for needle in [
            b"".as_slice(),      // Empty set: every membership bit is clear.
            b"baba",             // Repeated, unordered input for the same two letters.
            &[0, 127, 128, 255], // First and last byte of each table half.
            &diagonal,           // Sparse set with equal high and low nibbles.
            &complement,         // Dense set containing all other bytes.
            &all,                // Full set: every byte belongs to the needle.
        ] {
            visit(needle);
        }
    }

    #[test]
    fn needle_yields_bytes_in_ascending_order() {
        let table = TruffleTable::new(&[255, b'b', 0, b'A', 128]);
        assert!(table.needle().eq([0, b'A', b'b', 128, 255]));
    }

    #[test]
    fn needle_omits_duplicate_input_bytes() {
        let table = TruffleTable::new(b"aaaa");
        assert!(table.needle().eq(*b"a"));
    }

    #[test]
    fn search_reads_both_table_halves() {
        let mut lower = [0; 16];
        let mut upper = [0; 16];
        lower[0] = 0b00000001;
        lower[15] = 0b10000000;
        upper[0] = 0b00000001;
        upper[15] = 0b10000000;
        let table = TruffleTable { lower, upper };
        for byte in 0..=255 {
            assert_eq!(table.search(byte), [0, 127, 128, 255].contains(&byte));
        }
    }
}
