// ============================================================================
// world/types.rs — GPU uniform structs and buffer snapshot
// ============================================================================

use bytemuck::{Pod, Zeroable};

// ======================== Uniform Structs ========================

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SimParams {
    pub width: u32,
    pub height: u32,
    pub frame: u32,
    pub dt: f32,
    pub mutation_rate_mult: f32,
    pub predation_factor: f32,
    pub radius_cost_exp: f32,
    pub agg_mobility: f32,
    pub starvation_severity: f32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct VelocityParams {
    pub width: u32,
    pub height: u32,
    pub frame: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ResourceParams {
    pub width: u32,
    pub height: u32,
    pub diffusion: f32,
    pub feed_rate: f32,
    pub consumption: f32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct NormalizeParams {
    pub width: u32,
    pub height: u32,
    pub target_mass_x1000: u32,
    pub damping_x1000: u32,
    pub enabled: u32,
    pub dust_floor_x1000: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct RenderParams {
    pub width: u32,
    pub height: u32,
    pub visualization_mode: u32,
    pub show_legend: u32,
    pub time: f32,
    pub _pad: f32,
}

// ======================== Buffer Snapshot ========================

/// Raw CPU-side snapshot of simulation buffers (obtained via GPU readback).
pub struct BufferSnapshot {
    pub mass: Vec<f32>,
    pub energy: Vec<f32>,
    pub genome_a: Vec<f32>,
    pub genome_b: Vec<f32>,
    pub resource: Vec<f32>,
}
