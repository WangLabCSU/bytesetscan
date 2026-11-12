mod arch;
mod automatic;
mod explicit;
mod table;

pub use self::automatic::{Shufti, ShuftiFinder, ShuftiMismatcher};
pub use self::explicit::{ShuftiEngine, ShuftiEngineFinder, ShuftiEngineMismatcher};

use self::table::ShuftiTable;
