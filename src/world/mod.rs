// ============================================================================
// world/mod.rs — EvoLenia v2
// Re-exports all world-related types and functions.
// ============================================================================

mod constants;
pub mod init;
pub mod state;
pub mod types;

pub use constants::*;
pub use init::WorldState;
pub use types::*;
