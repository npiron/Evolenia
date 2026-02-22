// ============================================================================
// config.rs — EvoLenia v2
// Simulation parameters, visualization modes, and runtime configuration.
// Extended for Research Lab: all parameters exposed in the egui UI.
// ============================================================================

use serde::{Deserialize, Serialize};

/// Runtime simulation parameters adjustable via the Research Lab UI.
/// Every field here is wired to either a GPU uniform or engine state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulationParams {
    // -- Control --
    pub paused: bool,
    pub simulation_speed: u32,
    pub time_step: f32,
    pub vsync: bool,

    // -- Visualization --
    pub visualization_mode: u32,
    pub show_extended_ui: bool,

    // -- Evolution / Mutation --
    pub mutation_rate: f32,

    // -- Predation --
    pub predation_factor: f32,

    // -- Resources (Gray-Scott) --
    pub resource_diffusion: f32,
    pub resource_feed_rate: f32,
    pub resource_consumption: f32,

    // -- Mass normalization --
    pub mass_normalization_enabled: bool,
    pub mass_damping: f32,
    pub target_mass_multiplier: f32,

    // -- Non-linear trade-offs --
    pub radius_cost_exponent: f32,   // exponent for radius metabolic cost (1.0=linear, 2.0=quadratic)
    pub agg_mobility_tradeoff: f32,  // high agg reduces effective perception (0=disabled, 1=max)
    pub starvation_severity: f32,    // mass decay rate when energy depleted

    // -- Perturbations --
    pub perturbation_type: PerturbationType,
    pub perturbation_intensity: f32,   // 0.0–1.0 amplitude
    pub perturbation_radius: f32,      // spatial extent (fraction of world, 0.05–0.5)
    pub perturbation_active: bool,     // fire once (auto-clears)
    pub perturbation_center_x: f32,    // center in world-space [0,1]
    pub perturbation_center_y: f32,

    // -- Initial conditions (applied on restart) --
    pub num_seed_clusters: u32,
    pub seed_cluster_size: f32,
    pub initial_mass_fill: f32,

    // -- Reproducibility --
    pub seed: Option<u64>,
    pub use_fixed_seed: bool,
    pub fixed_seed_value: u64,
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            paused: false,
            simulation_speed: 1,
            time_step: 1.0,
            vsync: false,

            visualization_mode: 0,
            show_extended_ui: false,

            mutation_rate: 1.0,
            predation_factor: 1.0,

            resource_diffusion: 0.08,
            resource_feed_rate: 0.010,
            resource_consumption: 0.08,

            mass_normalization_enabled: true,
            mass_damping: 0.3,
            target_mass_multiplier: 1.0,

            radius_cost_exponent: 1.5,
            agg_mobility_tradeoff: 0.3,
            starvation_severity: 0.05,

            perturbation_type: PerturbationType::None,
            perturbation_intensity: 0.5,
            perturbation_radius: 0.15,
            perturbation_active: false,
            perturbation_center_x: 0.5,
            perturbation_center_y: 0.5,

            num_seed_clusters: 30,
            seed_cluster_size: 1.0,
            initial_mass_fill: 0.15,

            seed: None,
            use_fixed_seed: false,
            fixed_seed_value: 42,
        }
    }
}

impl SimulationParams {
    /// Compute the effective seed for reproducibility.
    pub fn effective_seed(&self) -> Option<u64> {
        if self.use_fixed_seed {
            Some(self.fixed_seed_value)
        } else {
            self.seed
        }
    }
}

/// Perturbation types for ecological experiments.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PerturbationType {
    None,
    Drought,         // reduces resources in area
    NutrientPulse,   // boosts resources in area
    MassStorm,       // randomizes mass/energy in area (catastrophe)
    MutationBurst,   // locally amplifies mutation rate
}

impl PerturbationType {
    pub fn all() -> &'static [PerturbationType] {
        &[
            PerturbationType::None,
            PerturbationType::Drought,
            PerturbationType::NutrientPulse,
            PerturbationType::MassStorm,
            PerturbationType::MutationBurst,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            PerturbationType::None => "None",
            PerturbationType::Drought => "Drought",
            PerturbationType::NutrientPulse => "Nutrient Pulse",
            PerturbationType::MassStorm => "Mass Storm",
            PerturbationType::MutationBurst => "Mutation Burst",
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
        5 => "Metabolic Stress",
        6 => "Advection Flux",
        7 => "Trophic Roles",
        _ => "Unknown",
    }
}

/// Total number of visualization modes available.
pub const VIS_MODE_COUNT: u32 = 8;
