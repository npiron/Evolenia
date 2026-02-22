// ============================================================================
// config.rs — EvoLenia v2
// Simulation parameters, visualization modes, and runtime configuration.
// ============================================================================

/// Runtime simulation parameters adjustable via keyboard.
#[derive(Clone, Debug)]
pub struct SimulationParams {
    pub paused: bool,
    pub visualization_mode: u32,
    pub show_extended_ui: bool,
    pub time_step: f32,
    pub mutation_rate: f32,
    pub simulation_speed: u32,
    pub vsync: bool,
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            paused: false,
            visualization_mode: 0,
            show_extended_ui: false,
            time_step: 1.0,
            mutation_rate: 1.0,
            simulation_speed: 1,
            vsync: false,
        }
    }
}

/// Returns the display name for a given visualization mode index.
pub fn visualization_mode_name(mode: u32) -> &'static str {
    match mode {
        0 => "Species Color",
        1 => "Energy Heatmap",
        2 => "Mass Density",
        3 => "Genetic Diversity",
        4 => "Predator/Prey",
        _ => "Unknown",
    }
}
