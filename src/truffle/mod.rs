//! Truffle checks byte-set membership with a 16-by-16 bitmap addressed by
//! nibbles, using SIMD table lookups. It is used in [Hyperscan]. Wojciech Muła
//! explains this representation in [*SIMDized check which bytes are in a set*][byte-lookup]
//! (2018), under “Universal algorithm”. Its “Alternative implementation”,
//! credited to Geoff Langdale, uses the shuffle instruction's zeroing behavior
//! to select between the two halves of the bitmap.
//!
//! [Hyperscan]: https://github.com/intel/hyperscan
//! [byte-lookup]: http://0x80.pl/notesen/2018-10-18-simd-byte-lookup.html#universal-algorithm
//!
//! # Matchers
//!
//! [`Truffle`] automatically selects the best supported backend for the current
//! environment. Use [`TruffleEngine`] to choose a specific
//! [`SimdBackend`](crate::SimdBackend); its constructor returns `None` if that
//! backend is unavailable.
//!
//! Both types provide `find` and `rfind` for bytes in the needle, and `mismatch`
//! and `rmismatch` for bytes outside it. Their `finder` and `mismatcher`
//! iterators support traversal in either direction. [`Truffle::needle`] returns
//! the matching byte set in ascending order.
//!
//! # How Truffle works
//!
//! Truffle represents every possible byte set in 32 bytes, with no grouping limit.
//! It divides the set into two 16-byte masks: one for values below `128` and one
//! for values from `128` through `255`.
//!
//! For a byte `b`, its top bit selects the mask. The low nibble selects an entry,
//! and bits 4 through 6 select a bit within that entry. A set bit means a match.
//! For the needle `[0x41, 0x61, 0xC1]`, only two entries are nonzero:
//!
//! | Entry      | Mask         | Accepted bytes |
//! | ---------- | ------------ | -------------- |
//! | `lower[1]` | `0b01010000` | `0x41`, `0x61` |
//! | `upper[1]` | `0b00010000` | `0xC1`         |
//!
//! For example, `0xC1` selects `upper[1]` and tests bit 4, which is set. `0x81`
//! selects the same entry but tests bit 0, which is clear. Each of the 256 possible
//! byte values has its own bit, so arbitrary sets are represented exactly.
//!
//! SIMD applies these lookups to several bytes at once. Each input byte still
//! selects its own table entry and membership bit.

mod arch;
mod automatic;
mod explicit;
mod table;

pub use self::automatic::{Truffle, TruffleFinder, TruffleMismatcher};
pub use self::explicit::{TruffleEngine, TruffleEngineFinder, TruffleEngineMismatcher};

use self::table::TruffleTable;
