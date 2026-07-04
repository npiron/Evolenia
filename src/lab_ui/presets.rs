// ============================================================================
// lab_ui/presets.rs — Preset catalog, categories, save/load
// ============================================================================

use crate::config::{PerturbationType, SimulationParams};

/// A preset with name, description, and parameters
pub struct PresetInfo {
    pub name: &'static str,
    pub emoji: &'static str,
    pub description: &'static str,
    pub category: &'static str,
    pub params: fn() -> SimulationParams,
}

/// Built-in preset catalog — all the interesting configurations.
pub fn preset_catalog() -> Vec<PresetInfo> {
    vec![
        // ============ AUTONOMY & MOVEMENT ============
        PresetInfo {
            name: "Lenia Gliders", emoji: "🛸",
            description: "Classic Lenia creatures (orbium, geminium). Self-propelled organisms that glide gracefully across the world.",
            category: "Autonomous",
            params: || SimulationParams {
                paused: false, simulation_speed: 1, time_step: 0.8, vsync: false,
                visualization_mode: 0, show_extended_ui: false,
                mutation_rate: 0.15, predation_factor: 0.5,
                resource_diffusion: 0.08, resource_feed_rate: 0.012, resource_consumption: 0.045,
                mass_normalization_enabled: true, mass_damping: 0.4, target_mass_multiplier: 0.85,
                radius_cost_exponent: 1.0, agg_mobility_tradeoff: 0.05, starvation_severity: 0.01,
                perturbation_type: PerturbationType::None, perturbation_intensity: 0.5, perturbation_radius: 0.15,
                perturbation_active: false, perturbation_center_x: 0.5, perturbation_center_y: 0.5,
                num_seed_clusters: 25, seed_cluster_size: 1.8, initial_mass_fill: 0.22,
                seed: None, use_fixed_seed: false, fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Swarm Drones", emoji: "🐝",
            description: "Fast-moving swarms of small entities. High mobility, coordinated movements, emergent flocking behavior.",
            category: "Autonomous",
            params: || SimulationParams {
                paused: false, simulation_speed: 1, time_step: 1.0, vsync: false,
                visualization_mode: 0, show_extended_ui: false,
                mutation_rate: 0.8, predation_factor: 1.5,
                resource_diffusion: 0.12, resource_feed_rate: 0.018, resource_consumption: 0.04,
                mass_normalization_enabled: false, mass_damping: 0.15, target_mass_multiplier: 1.2,
                radius_cost_exponent: 1.1, agg_mobility_tradeoff: 0.15, starvation_severity: 0.02,
                perturbation_type: PerturbationType::None, perturbation_intensity: 0.5, perturbation_radius: 0.15,
                perturbation_active: false, perturbation_center_x: 0.5, perturbation_center_y: 0.5,
                num_seed_clusters: 50, seed_cluster_size: 0.8, initial_mass_fill: 0.12,
                seed: None, use_fixed_seed: false, fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Wandering Blobs", emoji: "🫧",
            description: "Large slow-moving organisms with low mutation. They drift and pulse like jellyfish.",
            category: "Autonomous",
            params: || SimulationParams {
                paused: false, simulation_speed: 1, time_step: 0.6, vsync: false,
                visualization_mode: 0, show_extended_ui: false,
                mutation_rate: 0.1, predation_factor: 0.3,
                resource_diffusion: 0.06, resource_feed_rate: 0.014, resource_consumption: 0.035,
                mass_normalization_enabled: true, mass_damping: 0.5, target_mass_multiplier: 0.7,
                radius_cost_exponent: 0.9, agg_mobility_tradeoff: 0.0, starvation_severity: 0.005,
                perturbation_type: PerturbationType::None, perturbation_intensity: 0.5, perturbation_radius: 0.15,
                perturbation_active: false, perturbation_center_x: 0.5, perturbation_center_y: 0.5,
                num_seed_clusters: 15, seed_cluster_size: 2.5, initial_mass_fill: 0.18,
                seed: None, use_fixed_seed: false, fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Micro Rockets", emoji: "🚀",
            description: "Tiny fast entities that shoot across the screen. High speed, rapid evolution.",
            category: "Autonomous",
            params: || SimulationParams {
                paused: false, simulation_speed: 1, time_step: 1.2, vsync: false,
                visualization_mode: 0, show_extended_ui: false,
                mutation_rate: 1.5, predation_factor: 0.8,
                resource_diffusion: 0.15, resource_feed_rate: 0.022, resource_consumption: 0.03,
                mass_normalization_enabled: false, mass_damping: 0.1, target_mass_multiplier: 1.5,
                radius_cost_exponent: 0.8, agg_mobility_tradeoff: 0.1, starvation_severity: 0.015,
                perturbation_type: PerturbationType::None, perturbation_intensity: 0.5, perturbation_radius: 0.15,
                perturbation_active: false, perturbation_center_x: 0.5, perturbation_center_y: 0.5,
                num_seed_clusters: 80, seed_cluster_size: 0.5, initial_mass_fill: 0.08,
                seed: None, use_fixed_seed: false, fixed_seed_value: 42,
            },
        },
        // ============ PREDATION & COMPETITION ============
        PresetInfo {
            name: "Predator Prey", emoji: "🦁",
            description: "Intense hunter-prey dynamics. Predators chase, prey flee. Watch the ecosystem balance!",
            category: "Predation",
            params: || SimulationParams {
                paused: false, simulation_speed: 1, time_step: 1.0, vsync: false,
                visualization_mode: 0, show_extended_ui: false,
                mutation_rate: 0.5, predation_factor: 2.5,
                resource_diffusion: 0.07, resource_feed_rate: 0.016, resource_consumption: 0.07,
                mass_normalization_enabled: true, mass_damping: 0.3, target_mass_multiplier: 1.0,
                radius_cost_exponent: 1.4, agg_mobility_tradeoff: 0.6, starvation_severity: 0.04,
                perturbation_type: PerturbationType::None, perturbation_intensity: 0.5, perturbation_radius: 0.15,
                perturbation_active: false, perturbation_center_x: 0.5, perturbation_center_y: 0.5,
                num_seed_clusters: 35, seed_cluster_size: 1.2, initial_mass_fill: 0.20,
                seed: None, use_fixed_seed: false, fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Arms Race", emoji: "⚔️",
            description: "High mutation + high predation = constant evolutionary warfare. Species rise and fall.",
            category: "Predation",
            params: || SimulationParams {
                paused: false, simulation_speed: 1, time_step: 1.1, vsync: false,
                visualization_mode: 0, show_extended_ui: false,
                mutation_rate: 2.0, predation_factor: 3.0,
                resource_diffusion: 0.08, resource_feed_rate: 0.020, resource_consumption: 0.08,
                mass_normalization_enabled: true, mass_damping: 0.25, target_mass_multiplier: 1.1,
                radius_cost_exponent: 1.5, agg_mobility_tradeoff: 0.7, starvation_severity: 0.06,
                perturbation_type: PerturbationType::None, perturbation_intensity: 0.5, perturbation_radius: 0.15,
                perturbation_active: false, perturbation_center_x: 0.5, perturbation_center_y: 0.5,
                num_seed_clusters: 40, seed_cluster_size: 1.0, initial_mass_fill: 0.18,
                seed: None, use_fixed_seed: false, fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Apex Hunters", emoji: "🦈",
            description: "Few powerful predators dominate. Sparse resources, harsh survival.",
            category: "Predation",
            params: || SimulationParams {
                paused: false, simulation_speed: 1, time_step: 0.9, vsync: false,
                visualization_mode: 0, show_extended_ui: false,
                mutation_rate: 0.3, predation_factor: 4.0,
                resource_diffusion: 0.05, resource_feed_rate: 0.008, resource_consumption: 0.10,
                mass_normalization_enabled: true, mass_damping: 0.4, target_mass_multiplier: 0.8,
                radius_cost_exponent: 1.6, agg_mobility_tradeoff: 0.8, starvation_severity: 0.08,
                perturbation_type: PerturbationType::None, perturbation_intensity: 0.5, perturbation_radius: 0.15,
                perturbation_active: false, perturbation_center_x: 0.5, perturbation_center_y: 0.5,
                num_seed_clusters: 20, seed_cluster_size: 1.5, initial_mass_fill: 0.15,
                seed: None, use_fixed_seed: false, fixed_seed_value: 42,
            },
        },
        // ============ EXPLOSIVE & CHAOTIC ============
        PresetInfo {
            name: "Big Bang", emoji: "💥",
            description: "Explosive growth from few seeds. Watch life rapidly colonize the entire world!",
            category: "Explosive",
            params: || SimulationParams {
                paused: false, simulation_speed: 1, time_step: 1.3, vsync: false,
                visualization_mode: 0, show_extended_ui: false,
                mutation_rate: 2.5, predation_factor: 1.0,
                resource_diffusion: 0.18, resource_feed_rate: 0.030, resource_consumption: 0.025,
                mass_normalization_enabled: false, mass_damping: 0.05, target_mass_multiplier: 2.0,
                radius_cost_exponent: 0.7, agg_mobility_tradeoff: 0.1, starvation_severity: 0.008,
                perturbation_type: PerturbationType::None, perturbation_intensity: 0.5, perturbation_radius: 0.15,
                perturbation_active: false, perturbation_center_x: 0.5, perturbation_center_y: 0.5,
                num_seed_clusters: 5, seed_cluster_size: 3.0, initial_mass_fill: 0.05,
                seed: None, use_fixed_seed: false, fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Chaos Storm", emoji: "🌪️",
            description: "Maximum entropy! Wild mutations, unstable dynamics, kaleidoscopic patterns.",
            category: "Explosive",
            params: || SimulationParams {
                paused: false, simulation_speed: 2, time_step: 1.5, vsync: false,
                visualization_mode: 0, show_extended_ui: false,
                mutation_rate: 5.0, predation_factor: 2.0,
                resource_diffusion: 0.20, resource_feed_rate: 0.025, resource_consumption: 0.05,
                mass_normalization_enabled: false, mass_damping: 0.02, target_mass_multiplier: 1.8,
                radius_cost_exponent: 0.6, agg_mobility_tradeoff: 0.3, starvation_severity: 0.01,
                perturbation_type: PerturbationType::None, perturbation_intensity: 0.5, perturbation_radius: 0.15,
                perturbation_active: false, perturbation_center_x: 0.5, perturbation_center_y: 0.5,
                num_seed_clusters: 60, seed_cluster_size: 0.7, initial_mass_fill: 0.15,
                seed: None, use_fixed_seed: false, fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Nova Burst", emoji: "⭐",
            description: "Periodic explosions of life followed by collapses. Boom-bust cycles.",
            category: "Explosive",
            params: || SimulationParams {
                paused: false, simulation_speed: 1, time_step: 1.2, vsync: false,
                visualization_mode: 0, show_extended_ui: false,
                mutation_rate: 1.8, predation_factor: 1.5,
                resource_diffusion: 0.10, resource_feed_rate: 0.035, resource_consumption: 0.06,
                mass_normalization_enabled: true, mass_damping: 0.2, target_mass_multiplier: 1.3,
                radius_cost_exponent: 1.0, agg_mobility_tradeoff: 0.2, starvation_severity: 0.03,
                perturbation_type: PerturbationType::None, perturbation_intensity: 0.5, perturbation_radius: 0.15,
                perturbation_active: false, perturbation_center_x: 0.5, perturbation_center_y: 0.5,
                num_seed_clusters: 30, seed_cluster_size: 1.2, initial_mass_fill: 0.12,
                seed: None, use_fixed_seed: false, fixed_seed_value: 42,
            },
        },
        // ============ STABLE & AESTHETIC ============
        PresetInfo {
            name: "Zen Garden", emoji: "🪷",
            description: "Peaceful, stable patterns. Slow evolution, beautiful organic shapes.",
            category: "Stable",
            params: || SimulationParams {
                paused: false, simulation_speed: 1, time_step: 0.5, vsync: false,
                visualization_mode: 0, show_extended_ui: false,
                mutation_rate: 0.05, predation_factor: 0.2,
                resource_diffusion: 0.04, resource_feed_rate: 0.010, resource_consumption: 0.03,
                mass_normalization_enabled: true, mass_damping: 0.6, target_mass_multiplier: 0.6,
                radius_cost_exponent: 1.2, agg_mobility_tradeoff: 0.0, starvation_severity: 0.003,
                perturbation_type: PerturbationType::None, perturbation_intensity: 0.5, perturbation_radius: 0.15,
                perturbation_active: false, perturbation_center_x: 0.5, perturbation_center_y: 0.5,
                num_seed_clusters: 40, seed_cluster_size: 1.5, initial_mass_fill: 0.25,
                seed: None, use_fixed_seed: false, fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Coral Reef", emoji: "🪸",
            description: "Dense, colorful colonies. Slow-growing structures with high biodiversity.",
            category: "Stable",
            params: || SimulationParams {
                paused: false, simulation_speed: 1, time_step: 0.7, vsync: false,
                visualization_mode: 0, show_extended_ui: false,
                mutation_rate: 0.2, predation_factor: 0.4,
                resource_diffusion: 0.06, resource_feed_rate: 0.014, resource_consumption: 0.04,
                mass_normalization_enabled: true, mass_damping: 0.45, target_mass_multiplier: 0.75,
                radius_cost_exponent: 1.1, agg_mobility_tradeoff: 0.1, starvation_severity: 0.01,
                perturbation_type: PerturbationType::None, perturbation_intensity: 0.5, perturbation_radius: 0.15,
                perturbation_active: false, perturbation_center_x: 0.5, perturbation_center_y: 0.5,
                num_seed_clusters: 70, seed_cluster_size: 0.8, initial_mass_fill: 0.28,
                seed: None, use_fixed_seed: false, fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Crystal Growth", emoji: "💎",
            description: "Geometric, ordered growth patterns. Low chaos, high structure.",
            category: "Stable",
            params: || SimulationParams {
                paused: false, simulation_speed: 1, time_step: 0.4, vsync: false,
                visualization_mode: 0, show_extended_ui: false,
                mutation_rate: 0.02, predation_factor: 0.1,
                resource_diffusion: 0.03, resource_feed_rate: 0.008, resource_consumption: 0.025,
                mass_normalization_enabled: true, mass_damping: 0.7, target_mass_multiplier: 0.5,
                radius_cost_exponent: 1.3, agg_mobility_tradeoff: 0.0, starvation_severity: 0.002,
                perturbation_type: PerturbationType::None, perturbation_intensity: 0.5, perturbation_radius: 0.15,
                perturbation_active: false, perturbation_center_x: 0.5, perturbation_center_y: 0.5,
                num_seed_clusters: 30, seed_cluster_size: 2.0, initial_mass_fill: 0.30,
                seed: None, use_fixed_seed: false, fixed_seed_value: 42,
            },
        },
        // ============ SPECIAL EFFECTS ============
        PresetInfo {
            name: "Fireflies", emoji: "🔥",
            description: "Sparse glowing entities that flicker and dance in the dark.",
            category: "Special",
            params: || SimulationParams {
                paused: false, simulation_speed: 1, time_step: 0.9, vsync: false,
                visualization_mode: 0, show_extended_ui: false,
                mutation_rate: 0.6, predation_factor: 0.6,
                resource_diffusion: 0.09, resource_feed_rate: 0.011, resource_consumption: 0.055,
                mass_normalization_enabled: true, mass_damping: 0.35, target_mass_multiplier: 0.9,
                radius_cost_exponent: 1.0, agg_mobility_tradeoff: 0.2, starvation_severity: 0.02,
                perturbation_type: PerturbationType::None, perturbation_intensity: 0.5, perturbation_radius: 0.15,
                perturbation_active: false, perturbation_center_x: 0.5, perturbation_center_y: 0.5,
                num_seed_clusters: 100, seed_cluster_size: 0.4, initial_mass_fill: 0.06,
                seed: None, use_fixed_seed: false, fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Lava Flow", emoji: "🌋",
            description: "Slow-spreading mass that flows and pools. Use Thermal visualization mode!",
            category: "Special",
            params: || SimulationParams {
                paused: false, simulation_speed: 1, time_step: 0.6, vsync: false,
                visualization_mode: 2, show_extended_ui: false,
                mutation_rate: 0.3, predation_factor: 0.7,
                resource_diffusion: 0.15, resource_feed_rate: 0.020, resource_consumption: 0.02,
                mass_normalization_enabled: false, mass_damping: 0.08, target_mass_multiplier: 1.6,
                radius_cost_exponent: 0.8, agg_mobility_tradeoff: 0.05, starvation_severity: 0.005,
                perturbation_type: PerturbationType::None, perturbation_intensity: 0.5, perturbation_radius: 0.15,
                perturbation_active: false, perturbation_center_x: 0.5, perturbation_center_y: 0.5,
                num_seed_clusters: 10, seed_cluster_size: 3.5, initial_mass_fill: 0.10,
                seed: None, use_fixed_seed: false, fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Neural Network", emoji: "🧠",
            description: "Interconnected pathways that form and dissolve. Brain-like connectivity.",
            category: "Special",
            params: || SimulationParams {
                paused: false, simulation_speed: 1, time_step: 0.85, vsync: false,
                visualization_mode: 0, show_extended_ui: false,
                mutation_rate: 0.4, predation_factor: 0.5,
                resource_diffusion: 0.07, resource_feed_rate: 0.013, resource_consumption: 0.05,
                mass_normalization_enabled: true, mass_damping: 0.38, target_mass_multiplier: 0.85,
                radius_cost_exponent: 1.15, agg_mobility_tradeoff: 0.15, starvation_severity: 0.015,
                perturbation_type: PerturbationType::None, perturbation_intensity: 0.5, perturbation_radius: 0.15,
                perturbation_active: false, perturbation_center_x: 0.5, perturbation_center_y: 0.5,
                num_seed_clusters: 45, seed_cluster_size: 1.1, initial_mass_fill: 0.20,
                seed: None, use_fixed_seed: false, fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Galaxy Formation", emoji: "🌌",
            description: "Spiral arms emerge from rotating mass concentrations. Cosmic scale!",
            category: "Special",
            params: || SimulationParams {
                paused: false, simulation_speed: 1, time_step: 0.75, vsync: false,
                visualization_mode: 0, show_extended_ui: false,
                mutation_rate: 0.25, predation_factor: 0.8,
                resource_diffusion: 0.11, resource_feed_rate: 0.016, resource_consumption: 0.038,
                mass_normalization_enabled: true, mass_damping: 0.3, target_mass_multiplier: 1.0,
                radius_cost_exponent: 1.05, agg_mobility_tradeoff: 0.25, starvation_severity: 0.012,
                perturbation_type: PerturbationType::None, perturbation_intensity: 0.5, perturbation_radius: 0.15,
                perturbation_active: false, perturbation_center_x: 0.5, perturbation_center_y: 0.5,
                num_seed_clusters: 8, seed_cluster_size: 4.0, initial_mass_fill: 0.08,
                seed: None, use_fixed_seed: false, fixed_seed_value: 42,
            },
        },
        // ============ EXPERIMENTAL ============
        PresetInfo {
            name: "Evolution Lab", emoji: "🧬",
            description: "Balanced parameters for observing long-term evolutionary dynamics.",
            category: "Experimental",
            params: || SimulationParams {
                paused: false, simulation_speed: 1, time_step: 1.0, vsync: false,
                visualization_mode: 0, show_extended_ui: false,
                mutation_rate: 0.5, predation_factor: 1.0,
                resource_diffusion: 0.08, resource_feed_rate: 0.012, resource_consumption: 0.06,
                mass_normalization_enabled: true, mass_damping: 0.3, target_mass_multiplier: 1.0,
                radius_cost_exponent: 1.3, agg_mobility_tradeoff: 0.3, starvation_severity: 0.03,
                perturbation_type: PerturbationType::None, perturbation_intensity: 0.5, perturbation_radius: 0.15,
                perturbation_active: false, perturbation_center_x: 0.5, perturbation_center_y: 0.5,
                num_seed_clusters: 30, seed_cluster_size: 1.0, initial_mass_fill: 0.15,
                seed: None, use_fixed_seed: false, fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Patch Mosaic", emoji: "🧩",
            description: "Patchy nutrient landscape with strong niche pressure. Favors local coexistence over global takeover.",
            category: "Experimental",
            params: || SimulationParams {
                paused: false, simulation_speed: 1, time_step: 0.9, vsync: false,
                visualization_mode: 0, show_extended_ui: false,
                mutation_rate: 0.45, predation_factor: 1.2,
                resource_diffusion: 0.045, resource_feed_rate: 0.010, resource_consumption: 0.075,
                mass_normalization_enabled: true, mass_damping: 0.42, target_mass_multiplier: 0.78,
                radius_cost_exponent: 1.6, agg_mobility_tradeoff: 0.45, starvation_severity: 0.05,
                perturbation_type: PerturbationType::None, perturbation_intensity: 0.5, perturbation_radius: 0.15,
                perturbation_active: false, perturbation_center_x: 0.5, perturbation_center_y: 0.5,
                num_seed_clusters: 55, seed_cluster_size: 0.9, initial_mass_fill: 0.14,
                seed: None, use_fixed_seed: false, fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Frontier Cycles", emoji: "🌊",
            description: "Traveling competition fronts with repeated boom-bust waves instead of static full coverage.",
            category: "Experimental",
            params: || SimulationParams {
                paused: false, simulation_speed: 1, time_step: 0.95, vsync: false,
                visualization_mode: 0, show_extended_ui: false,
                mutation_rate: 0.9, predation_factor: 1.8,
                resource_diffusion: 0.095, resource_feed_rate: 0.017, resource_consumption: 0.065,
                mass_normalization_enabled: true, mass_damping: 0.22, target_mass_multiplier: 1.05,
                radius_cost_exponent: 1.35, agg_mobility_tradeoff: 0.55, starvation_severity: 0.038,
                perturbation_type: PerturbationType::None, perturbation_intensity: 0.5, perturbation_radius: 0.15,
                perturbation_active: false, perturbation_center_x: 0.5, perturbation_center_y: 0.5,
                num_seed_clusters: 36, seed_cluster_size: 1.1, initial_mass_fill: 0.16,
                seed: None, use_fixed_seed: false, fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Island Succession", emoji: "🏞️",
            description: "Low-diffusion resource islands create local successions and refuges, slowing homogenization.",
            category: "Experimental",
            params: || SimulationParams {
                paused: false, simulation_speed: 1, time_step: 0.85, vsync: false,
                visualization_mode: 0, show_extended_ui: false,
                mutation_rate: 0.35, predation_factor: 0.9,
                resource_diffusion: 0.025, resource_feed_rate: 0.013, resource_consumption: 0.055,
                mass_normalization_enabled: true, mass_damping: 0.50, target_mass_multiplier: 0.72,
                radius_cost_exponent: 1.55, agg_mobility_tradeoff: 0.35, starvation_severity: 0.03,
                perturbation_type: PerturbationType::None, perturbation_intensity: 0.5, perturbation_radius: 0.15,
                perturbation_active: false, perturbation_center_x: 0.5, perturbation_center_y: 0.5,
                num_seed_clusters: 18, seed_cluster_size: 2.2, initial_mass_fill: 0.12,
                seed: None, use_fixed_seed: false, fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Sparse World", emoji: "🏜️",
            description: "Harsh desert conditions. Only the fittest survive in resource-poor lands.",
            category: "Experimental",
            params: || SimulationParams {
                paused: false, simulation_speed: 1, time_step: 0.8, vsync: false,
                visualization_mode: 0, show_extended_ui: false,
                mutation_rate: 0.4, predation_factor: 1.2,
                resource_diffusion: 0.04, resource_feed_rate: 0.006, resource_consumption: 0.08,
                mass_normalization_enabled: true, mass_damping: 0.5, target_mass_multiplier: 0.6,
                radius_cost_exponent: 1.5, agg_mobility_tradeoff: 0.4, starvation_severity: 0.06,
                perturbation_type: PerturbationType::None, perturbation_intensity: 0.5, perturbation_radius: 0.15,
                perturbation_active: false, perturbation_center_x: 0.5, perturbation_center_y: 0.5,
                num_seed_clusters: 15, seed_cluster_size: 1.0, initial_mass_fill: 0.10,
                seed: None, use_fixed_seed: false, fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Rich Paradise", emoji: "🏝️",
            description: "Abundant resources! Everything grows easily. Watch explosion of diversity.",
            category: "Experimental",
            params: || SimulationParams {
                paused: false, simulation_speed: 1, time_step: 1.1, vsync: false,
                visualization_mode: 0, show_extended_ui: false,
                mutation_rate: 1.0, predation_factor: 0.6,
                resource_diffusion: 0.12, resource_feed_rate: 0.028, resource_consumption: 0.025,
                mass_normalization_enabled: false, mass_damping: 0.1, target_mass_multiplier: 1.5,
                radius_cost_exponent: 0.85, agg_mobility_tradeoff: 0.1, starvation_severity: 0.005,
                perturbation_type: PerturbationType::None, perturbation_intensity: 0.5, perturbation_radius: 0.15,
                perturbation_active: false, perturbation_center_x: 0.5, perturbation_center_y: 0.5,
                num_seed_clusters: 25, seed_cluster_size: 1.5, initial_mass_fill: 0.12,
                seed: None, use_fixed_seed: false, fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Speed Demon", emoji: "⚡",
            description: "Maximum simulation speed! Fast time step, rapid evolution cycles.",
            category: "Experimental",
            params: || SimulationParams {
                paused: false, simulation_speed: 3, time_step: 2.0, vsync: false,
                visualization_mode: 0, show_extended_ui: false,
                mutation_rate: 1.2, predation_factor: 1.0,
                resource_diffusion: 0.10, resource_feed_rate: 0.018, resource_consumption: 0.05,
                mass_normalization_enabled: true, mass_damping: 0.2, target_mass_multiplier: 1.2,
                radius_cost_exponent: 1.1, agg_mobility_tradeoff: 0.2, starvation_severity: 0.025,
                perturbation_type: PerturbationType::None, perturbation_intensity: 0.5, perturbation_radius: 0.15,
                perturbation_active: false, perturbation_center_x: 0.5, perturbation_center_y: 0.5,
                num_seed_clusters: 40, seed_cluster_size: 1.0, initial_mass_fill: 0.15,
                seed: None, use_fixed_seed: false, fixed_seed_value: 42,
            },
        },
    ]
}

/// Get unique categories from the catalog.
pub fn preset_categories() -> Vec<&'static str> {
    vec![
        "Autonomous",
        "Predation",
        "Explosive",
        "Stable",
        "Special",
        "Experimental",
    ]
}

// ======================== Preset Save/Load ========================

pub fn save_preset(name: &str, params: &SimulationParams) {
    let dir = std::path::PathBuf::from("presets");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        log::error!("Failed to create presets dir: {}", e);
        return;
    }
    let path = dir.join(format!("{}.json", name));
    match serde_json::to_string_pretty(params) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&path, json) {
                log::error!("Failed to save preset: {}", e);
            } else {
                log::info!("Preset saved: {:?}", path);
            }
        }
        Err(e) => log::error!("Failed to serialize preset: {}", e),
    }
}

pub fn load_preset(name: &str) -> Option<SimulationParams> {
    let path = std::path::PathBuf::from(format!("presets/{}.json", name));
    let content = std::fs::read_to_string(&path).ok()?;
    match serde_json::from_str::<SimulationParams>(&content) {
        Ok(params) => {
            log::info!("Loaded preset from {:?}", path);
            Some(params)
        }
        Err(e) => {
            log::error!("Failed to parse preset {:?}: {}", path, e);
            None
        }
    }
}
