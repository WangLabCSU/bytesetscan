//! Shufti checks byte-set membership with two nibble-indexed tables of group
//! flags. It is used in [Hyperscan]. Wojciech Muła explains the same table lookup
//! and intersection technique in [*SIMDized check which bytes are in a set*][byte-lookup]
//! (2018), under “Special case 1 — small sets”. That example assigns one flag
//! per element; Shufti can share a flag among several elements.
//!
//! Each table entry has eight bits, one flag per group. Any set of ASCII bytes
//! fits within these eight groups. Some sets containing non-ASCII bytes need
//! more groups; [`crate::truffle`] can represent those sets instead.
//!
//! [Hyperscan]: https://github.com/intel/hyperscan
//! [byte-lookup]: http://0x80.pl/notesen/2018-10-18-simd-byte-lookup.html#special-case-1-small-sets
//!
//! # Matchers
//!
//! [`Shufti`] automatically selects the best supported backend for the current
//! environment. Use [`ShuftiEngine`] to choose a specific
//! [`SimdBackend`](crate::SimdBackend); its constructor returns `None` if that
//! backend is unavailable.
//!
//! Both types provide `find` and `rfind` for bytes in the needle, and `mismatch`
//! and `rmismatch` for bytes outside it. Their `finder` and `mismatcher`
//! iterators support traversal in either direction. [`Shufti::needle`] returns
//! the matching byte set in ascending order.
//!
//! # How Shufti works
//!
//! Shufti encodes a byte set in two tables with 16 entries each. A byte has two
//! four-bit halves, called **nibbles**, each with a value from 0 to 15:
//!
//! ```text
//! byte b = (high nibble << 4) | low nibble
//!
//! low nibble  = b & 0b00001111
//! high nibble = b >> 4
//! ```
//!
//! Each table entry is a set of group flags. A byte belongs to the set when the
//! entries selected by its two nibbles share at least one flag. For `b"ABRabr"`,
//! the tables can be written as:
//!
//! ```rust
//! let mut low = [0u8; 16];
//! let mut high = [0u8; 16];
//! low[1] = 0b00000001;
//! low[2] = 0b00000011;
//! high[4] = 0b00000001;
//! high[5] = 0b00000010;
//! high[6] = 0b00000001;
//! high[7] = 0b00000010;
//!
//! let accepts = |byte: u8| {
//!     low[(byte & 0b00001111) as usize] & high[(byte >> 4) as usize] != 0
//! };
//!
//! assert!(accepts(b'B'));
//! assert!(accepts(b'R'));
//! assert!(!accepts(b'Q'));
//! ```
//!
//! The membership test uses two table lookups and a bitwise AND. The constructors
//! build these tables from the needle; the following example explains how.
//!
//! ## Building the groups: `ABRabr`
//!
//! Imagine a 16-by-16 grid of all possible byte values. The high nibble selects a
//! row, and the low nibble selects a column. Mark the cells for the bytes to accept.
//! For `b"ABRabr"`, the nonempty rows are:
//!
//! | High nibble | Accepted bytes             | Accepted low nibbles | Group flag   |
//! | ----------- | -------------------------- | -------------------- | ------------ |
//! | `0x4`       | `A` (`0x41`), `B` (`0x42`) | `{1, 2}`             | `0b00000001` |
//! | `0x5`       | `R` (`0x52`)               | `{2}`                | `0b00000010` |
//! | `0x6`       | `a` (`0x61`), `b` (`0x62`) | `{1, 2}`             | `0b00000001` |
//! | `0x7`       | `r` (`0x72`)               | `{2}`                | `0b00000010` |
//!
//! The constructor groups rows with identical sets of accepted columns. Rows
//! `0x4` and `0x6` share one flag because both accept `{1, 2}`. Rows `0x5` and
//! `0x7` share another because both accept only `{2}`. Empty rows need no flag.
//! Construction returns `None` if there are more than eight distinct nonempty
//! row patterns. This is the grouping strategy used by these constructors.
//!
//! The high table selects a row's group. The low table records every group that
//! accepts a particular column. The nonzero entries in this example are:
//!
//! | Table entries        | Flags        | Meaning                                    |
//! | -------------------- | ------------ | ------------------------------------------ |
//! | `high[4]`, `high[6]` | `0b00000001` | These rows accept low nibbles 1 and 2.     |
//! | `high[5]`, `high[7]` | `0b00000010` | These rows accept only low nibble 2.       |
//! | `low[1]`             | `0b00000001` | Only the first group accepts low nibble 1. |
//! | `low[2]`             | `0b00000011` | Both groups accept low nibble 2.           |
//!
//! All other entries are zero. The flags now distinguish bytes that share just
//! one nibble:
//!
//! | Byte         | Lookup                             | Result   |
//! | ------------ | ---------------------------------- | -------- |
//! | `B` (`0x42`) | `low[2] & high[4]` = `0b11 & 0b01` | Match    |
//! | `R` (`0x52`) | `low[2] & high[5]` = `0b11 & 0b10` | Match    |
//! | `Q` (`0x51`) | `low[1] & high[5]` = `0b01 & 0b10` | No match |
//!
//! Simply checking that both nibbles occur somewhere in the set would incorrectly
//! accept `Q`: its high nibble occurs in `R`, and its low nibble occurs in `A`.
//! The shared-flag test requires the two nibbles to belong to the same group.
//! Since rows share a group only when their accepted columns are identical, the
//! test accepts exactly the marked cells. No second verification step is needed.

mod arch;
mod automatic;
mod explicit;
mod table;

pub use self::automatic::{Shufti, ShuftiFinder, ShuftiMismatcher};
pub use self::explicit::{ShuftiEngine, ShuftiEngineFinder, ShuftiEngineMismatcher};

use self::table::ShuftiTable;
