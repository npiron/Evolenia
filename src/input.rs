// ============================================================================
// input.rs — EvoLenia v2
// Keyboard state tracking for continuous held-key actions.
// ============================================================================

/// Tracks which navigation keys are currently held down.
#[derive(Default)]
pub struct KeysHeld {
    pub w: bool,
    pub s: bool,
    pub a: bool,
    pub d: bool,
    pub q: bool,
    pub e: bool,
}
