mod arch;
mod automatic;
mod explicit;
mod table;

pub use self::automatic::{Bitset, BitsetFinder, BitsetMismatcher};
pub use self::explicit::{BitsetEngine, BitsetEngineFinder, BitsetEngineMismatcher};

use self::table::BitsetTable;
