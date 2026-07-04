// ============================================================================
// world/constants.rs — Grid size, workgroup layout, and mass target constants
// ============================================================================

// Performance tuning:
// - 512×512 = 4× faster than 1024×1024 (good for development/testing)
// - 1024×1024 = balanced (default, ~60 FPS on M1 Pro)
// - 2048×2048 = highest quality (requires powerful GPU)
pub const WORLD_WIDTH: u32 = 512;
pub const WORLD_HEIGHT: u32 = 512;
pub const WORKGROUP_X: u32 = 16;
pub const WORKGROUP_Y: u32 = 16;
pub const DT: f32 = 0.1; // reduced for stability
pub const TARGET_FILL: f32 = 0.15; // 15% initial mass fill

pub fn total_pixels() -> u32 {
    WORLD_WIDTH * WORLD_HEIGHT
}

pub fn target_total_mass() -> f32 {
    WORLD_WIDTH as f32 * WORLD_HEIGHT as f32 * TARGET_FILL
}
