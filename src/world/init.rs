// ============================================================================
// world/init.rs — WorldState struct and new_with_config initialization
// ============================================================================

use bytemuck;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use wgpu::util::DeviceExt;

use super::constants::*;
use super::types::*;
use crate::config::SimulationParams;

// ======================== WorldState ========================

pub struct WorldState {
    // Ping-pong buffer index: 0 or 1
    pub current: usize,

    // Mass buffers (ping-pong)
    pub mass: [wgpu::Buffer; 2],
    // Energy buffers (ping-pong)
    pub energy: [wgpu::Buffer; 2],
    // Genome A buffers: vec4(r, mu, sigma, aggressivity) (ping-pong)
    pub genome_a: [wgpu::Buffer; 2],
    // Genome B buffers: f32 mutation_rate (ping-pong)
    pub genome_b: [wgpu::Buffer; 2],

    // Single buffers (updated in-place)
    pub resource_map: wgpu::Buffer,
    pub velocity: wgpu::Buffer,

    // Atomic sum buffer for mass normalization
    pub mass_sum: wgpu::Buffer,

    // Staging buffers for CPU readback (diagnostics)
    pub staging_mass: wgpu::Buffer,
    pub staging_energy: wgpu::Buffer,
    pub staging_genome_a: wgpu::Buffer,
    pub staging_genome_b: wgpu::Buffer,
    pub staging_resource: wgpu::Buffer,

    // Uniform buffers
    pub sim_params_buffer: wgpu::Buffer,
    pub velocity_params_buffer: wgpu::Buffer,
    pub resource_params_buffer: wgpu::Buffer,
    pub normalize_params_buffer: wgpu::Buffer,
    pub render_params_buffer: wgpu::Buffer,

    pub frame: u32,
}

impl WorldState {
    /// Create a new world with default seed and parameters.
    pub fn new_with_config(
        device: &wgpu::Device,
        seed: Option<u64>,
        params: &SimulationParams,
    ) -> Self {
        let n = total_pixels() as usize;
        let mut rng: rand::rngs::StdRng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };

        let cluster_count_hint = params.num_seed_clusters.clamp(5, 100) as f32;
        let cluster_size_scale = params.seed_cluster_size.clamp(0.5, 3.0);
        let target_fill = params.initial_mass_fill.clamp(0.05, 0.5);

        // ---- Initialize data on CPU ----
        let mut mass_data = vec![0.0f32; n];
        let mut energy_data = vec![0.5f32; n];
        let mut genome_a_data = vec![[10.0f32, 0.15, 0.017, 0.0]; n];
        let mut genome_b_data = vec![0.003f32; n];
        let mut resource_data = vec![1.0f32; n];

        // ======================== Seed Patterns ========================
        let w = WORLD_WIDTH as i32;
        let h = WORLD_HEIGHT as i32;

        let pixel_idx = |px: i32, py: i32| -> usize {
            let wx = ((px % w) + w) % w;
            let wy = ((py % h) + h) % h;
            (wy as u32 * WORLD_WIDTH + wx as u32) as usize
        };

        let stamp = |mass: &mut [f32],
                     energy: &mut [f32],
                     ga: &mut [[f32; 4]],
                     gb: &mut [f32],
                     idx: usize,
                     m: f32,
                     e: f32,
                     genome: [f32; 4],
                     mut_rate: f32| {
            mass[idx] = (mass[idx] + m).min(1.0);
            energy[idx] = e;
            ga[idx] = genome;
            gb[idx] = mut_rate;
        };

        let random_genome = |rng: &mut rand::rngs::StdRng| -> ([f32; 4], f32) {
            let gene_r: f32 = rng.gen_range(7.0..14.0);
            let gene_mu: f32 = rng.gen_range(0.10..0.22);
            let gene_sigma: f32 = rng.gen_range(0.010..0.045);
            let gene_agg: f32 = rng.gen_range(0.0..0.4);
            let gene_mut: f32 = rng.gen_range(0.0003..0.004);
            ([gene_r, gene_mu, gene_sigma, gene_agg], gene_mut)
        };

        // ---- PATTERN 1: Gaussian clusters ----
        let num_clusters = ((cluster_count_hint * 0.42).round() as i32).max(4);
        for _ in 0..num_clusters {
            let cx = rng.gen_range(0..w);
            let cy = rng.gen_range(0..h);
            let radius = rng.gen_range(8.0..18.0) * cluster_size_scale;
            let (genome, mut_rate) = random_genome(&mut rng);
            let ir = radius as i32 + 1;
            for dy in -ir..=ir {
                for dx in -ir..=ir {
                    let dist = ((dx * dx + dy * dy) as f32).sqrt();
                    if dist > radius { continue; }
                    let falloff = (-dist * dist / (2.0 * radius * radius * 0.25)).exp();
                    let idx = pixel_idx(cx + dx, cy + dy);
                    stamp(&mut mass_data, &mut energy_data, &mut genome_a_data, &mut genome_b_data, idx, falloff, 0.5, genome, mut_rate);
                }
            }
        }

        // ---- PATTERN 2: Rings / annuli ----
        let num_rings = ((cluster_count_hint * 0.16).round() as i32).max(2);
        for _ in 0..num_rings {
            let cx = rng.gen_range(0..w);
            let cy = rng.gen_range(0..h);
            let outer_r = rng.gen_range(12.0..26.0) * cluster_size_scale;
            let inner_r = outer_r * rng.gen_range(0.4..0.7);
            let thickness = (outer_r - inner_r).max(2.0);
            let (genome, mut_rate) = random_genome(&mut rng);
            let ir = outer_r as i32 + 1;
            for dy in -ir..=ir {
                for dx in -ir..=ir {
                    let dist = ((dx * dx + dy * dy) as f32).sqrt();
                    if dist > outer_r || dist < inner_r { continue; }
                    let edge_outer = 1.0 - ((dist - outer_r + thickness * 0.3) / (thickness * 0.3)).max(0.0);
                    let edge_inner = ((dist - inner_r) / (thickness * 0.3)).min(1.0);
                    let m = (edge_outer * edge_inner).clamp(0.0, 1.0);
                    if m < 0.01 { continue; }
                    let idx = pixel_idx(cx + dx, cy + dy);
                    stamp(&mut mass_data, &mut energy_data, &mut genome_a_data, &mut genome_b_data, idx, m * 0.8, 0.6, genome, mut_rate);
                }
            }
        }

        // ---- PATTERN 3: Lines / filaments ----
        let num_lines = ((cluster_count_hint * 0.12).round() as i32).max(2);
        for _ in 0..num_lines {
            let x0 = rng.gen_range(0..w);
            let y0 = rng.gen_range(0..h);
            let angle: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
            let length = rng.gen_range(30.0..80.0) * cluster_size_scale;
            let half_width = rng.gen_range(1.5..4.0_f32) * cluster_size_scale;
            let (genome, mut_rate) = random_genome(&mut rng);
            let curvature: f32 = rng.gen_range(-0.02..0.02);
            let steps = (length * 2.0) as i32;
            for s in 0..=steps {
                let t = s as f32 / steps as f32;
                let a = angle + curvature * t * length;
                let lx = x0 as f32 + a.cos() * t * length;
                let ly = y0 as f32 + a.sin() * t * length;
                let hw = half_width as i32 + 1;
                for dy in -hw..=hw {
                    for dx in -hw..=hw {
                        let d = ((dx * dx + dy * dy) as f32).sqrt();
                        if d > half_width { continue; }
                        let m = (1.0 - d / half_width).max(0.0);
                        let idx = pixel_idx(lx as i32 + dx, ly as i32 + dy);
                        stamp(&mut mass_data, &mut energy_data, &mut genome_a_data, &mut genome_b_data, idx, m * 0.7, 0.5, genome, mut_rate);
                    }
                }
            }
        }

        // ---- PATTERN 4: Spirals ----
        let num_spirals = ((cluster_count_hint * 0.09).round() as i32).max(1);
        for _ in 0..num_spirals {
            let cx = rng.gen_range(0..w) as f32;
            let cy = rng.gen_range(0..h) as f32;
            let arms: u32 = rng.gen_range(2..5);
            let max_angle: f32 = rng.gen_range(3.0..6.0);
            let scale = rng.gen_range(15.0..35.0_f32) * cluster_size_scale;
            let arm_width = rng.gen_range(1.5..3.5_f32) * cluster_size_scale;
            let (genome, mut_rate) = random_genome(&mut rng);
            let steps = (max_angle * scale * 2.0) as i32;
            for arm in 0..arms {
                let arm_offset = std::f32::consts::TAU * arm as f32 / arms as f32;
                for s in 0..=steps {
                    let t = s as f32 / steps as f32;
                    let theta = t * max_angle + arm_offset;
                    let r = t * scale;
                    let sx = cx + theta.cos() * r;
                    let sy = cy + theta.sin() * r;
                    let hw = arm_width as i32 + 1;
                    for dy in -hw..=hw {
                        for dx in -hw..=hw {
                            let d = ((dx * dx + dy * dy) as f32).sqrt();
                            if d > arm_width { continue; }
                            let m = (1.0 - d / arm_width) * (1.0 - t * 0.3);
                            if m < 0.01 { continue; }
                            let idx = pixel_idx(sx as i32 + dx, sy as i32 + dy);
                            stamp(&mut mass_data, &mut energy_data, &mut genome_a_data, &mut genome_b_data, idx, m * 0.6, 0.55, genome, mut_rate);
                        }
                    }
                }
            }
        }

        // ---- PATTERN 5: Scattered noise patches ----
        let num_patches = ((cluster_count_hint * 0.20).round() as i32).max(3);
        for _ in 0..num_patches {
            let cx = rng.gen_range(0..w);
            let cy = rng.gen_range(0..h);
            let patch_r = (rng.gen_range(15.0..40.0) * cluster_size_scale) as i32;
            let density: f32 = rng.gen_range(0.05..0.15);
            let (genome, mut_rate) = random_genome(&mut rng);
            for dy in -patch_r..=patch_r {
                for dx in -patch_r..=patch_r {
                    let dist = ((dx * dx + dy * dy) as f32).sqrt();
                    if dist > patch_r as f32 { continue; }
                    if rng.gen::<f32>() > density { continue; }
                    let falloff = 1.0 - dist / patch_r as f32;
                    let m = falloff * rng.gen_range(0.1..0.5);
                    let idx = pixel_idx(cx + dx, cy + dy);
                    stamp(&mut mass_data, &mut energy_data, &mut genome_a_data, &mut genome_b_data, idx, m, 0.4, genome, mut_rate);
                }
            }
        }

        // ---- PATTERN 6: Apex predator nests ----
        let predation_bias = params.predation_factor.clamp(0.0, 3.0);
        let num_predators = (1.0 + predation_bias * 2.0).round() as i32;
        for _ in 0..num_predators {
            let cx = rng.gen_range(0..w);
            let cy = rng.gen_range(0..h);
            let radius = rng.gen_range(4.0..9.0) * cluster_size_scale;
            let gene_r: f32 = rng.gen_range(7.0..12.0);
            let gene_mu: f32 = rng.gen_range(0.12..0.22);
            let gene_sigma: f32 = rng.gen_range(0.015..0.040);
            let gene_agg: f32 = rng.gen_range(0.7..1.0);
            let gene_mut: f32 = rng.gen_range(0.001..0.004);
            let genome = [gene_r, gene_mu, gene_sigma, gene_agg];
            let ir = radius as i32 + 1;
            for dy in -ir..=ir {
                for dx in -ir..=ir {
                    let dist = ((dx * dx + dy * dy) as f32).sqrt();
                    if dist > radius { continue; }
                    let m = (-dist * dist / (2.0 * radius * radius * 0.3)).exp();
                    let idx = pixel_idx(cx + dx, cy + dy);
                    stamp(&mut mass_data, &mut energy_data, &mut genome_a_data, &mut genome_b_data, idx, m * 0.9, 0.8, genome, gene_mut);
                }
            }
        }

        // ---- PATTERN 7: Lenia Orbium seeds ----
        let lenia_creatures: Vec<([f32; 4], f32, f32, &str)> = vec![
            ([13.0, 0.15, 0.017, 0.0], 0.001, 13.0, "orbium"),
            ([13.0, 0.15, 0.017, 0.0], 0.001, 13.0, "orbium"),
            ([13.0, 0.14, 0.014, 0.0], 0.001, 12.0, "geminium"),
            ([14.0, 0.20, 0.030, 0.0], 0.001, 14.0, "scutium"),
            ([10.0, 0.13, 0.012, 0.0], 0.002, 10.0, "small_orbium"),
            ([12.0, 0.16, 0.020, 0.0], 0.001, 12.0, "orbium_var"),
            ([11.0, 0.18, 0.025, 0.0], 0.002, 11.0, "smooth_life"),
            ([15.0, 0.12, 0.013, 0.0], 0.001, 15.0, "large_orbium"),
        ];

        let lenia_bias = (1.0 - (params.mutation_rate / 5.0)).clamp(0.0, 1.0)
            * (1.0 - (params.predation_factor / 3.0)).clamp(0.0, 1.0);
        let desired_lenia = ((cluster_count_hint * 0.06) * (0.35 + lenia_bias * 0.85)).round() as usize;
        let lenia_count = desired_lenia.clamp(1, lenia_creatures.len());

        for (genome, mut_rate, pattern_r, _name) in lenia_creatures
            .choose_multiple(&mut rng, lenia_count)
            .copied()
        {
            let cx = rng.gen_range(20..(w - 20));
            let cy = rng.gen_range(20..(h - 20));
            let pr = pattern_r * cluster_size_scale;
            let ir = pr as i32 + 2;
            for dy in -ir..=ir {
                for dx in -ir..=ir {
                    let dist = ((dx * dx + dy * dy) as f32).sqrt();
                    if dist > pr { continue; }
                    let normalized = dist / pr;
                    let diff = normalized - 0.5;
                    let m = (-diff * diff / (2.0 * 0.15 * 0.15)).exp();
                    if m < 0.01 { continue; }
                    let idx = pixel_idx(cx + dx, cy + dy);
                    stamp(&mut mass_data, &mut energy_data, &mut genome_a_data, &mut genome_b_data, idx, m * 0.85, 0.7, genome, mut_rate);
                }
            }
            let asym_dx: i32 = rng.gen_range(-2..3);
            let asym_dy: i32 = rng.gen_range(-2..3);
            for d in 0..3 {
                let idx = pixel_idx(cx + asym_dx + d, cy + asym_dy);
                mass_data[idx] = (mass_data[idx] + 0.15).min(1.0);
            }
        }

        // ---- PATTERN 8: Lenia blob clusters ----
        let num_blobs = ((cluster_count_hint * 0.30).round() as i32).max(4);
        for _ in 0..num_blobs {
            let cx = rng.gen_range(0..w);
            let cy = rng.gen_range(0..h);
            let blob_r = rng.gen_range(8.0..16.0) * cluster_size_scale;
            let gene_r: f32 = rng.gen_range(9.0..14.0);
            let gene_mu: f32 = rng.gen_range(0.12..0.20);
            let gene_sigma: f32 = rng.gen_range(0.012..0.030);
            let gene_agg: f32 = 0.0;
            let gene_mut: f32 = rng.gen_range(0.001..0.003);
            let genome = [gene_r, gene_mu, gene_sigma, gene_agg];
            let ir = blob_r as i32 + 1;
            for dy in -ir..=ir {
                for dx in -ir..=ir {
                    let dist = ((dx * dx + dy * dy) as f32).sqrt();
                    if dist > blob_r { continue; }
                    let m = (-dist * dist / (2.0 * blob_r * blob_r * 0.2)).exp();
                    if m < 0.01 { continue; }
                    let idx = pixel_idx(cx + dx, cy + dy);
                    stamp(&mut mass_data, &mut energy_data, &mut genome_a_data, &mut genome_b_data, idx, m * 0.7, 0.6, genome, gene_mut);
                }
            }
        }

        // Match the requested initial fill from presets/UI
        let current_total_mass: f32 = mass_data.iter().sum();
        let requested_total_mass = WORLD_WIDTH as f32 * WORLD_HEIGHT as f32 * target_fill;
        if current_total_mass > 1e-6 {
            let scale = (requested_total_mass / current_total_mass).clamp(0.1, 4.0);
            for m in mass_data.iter_mut() {
                *m = (*m * scale).clamp(0.0, 1.0);
            }
        }

        // ======================== Resource Map Heterogeneity ========================
        let base_resource = (0.45 + params.resource_feed_rate * 9.0
            - params.resource_consumption * 1.8)
            .clamp(0.2, 0.9);
        for r in resource_data.iter_mut() {
            *r = base_resource;
        }

        let num_oases = ((cluster_count_hint * 0.20).round() as i32).max(3);
        let oasis_boost_strength = (0.18 + params.resource_feed_rate * 8.0).clamp(0.15, 0.45);
        for _ in 0..num_oases {
            let cx = rng.gen_range(0..w);
            let cy = rng.gen_range(0..h);
            let radius = rng.gen_range(20.0..60.0) * cluster_size_scale;
            let ir = radius as i32 + 1;
            for dy in -ir..=ir {
                for dx in -ir..=ir {
                    let dist = ((dx * dx + dy * dy) as f32).sqrt();
                    if dist > radius { continue; }
                    let boost = oasis_boost_strength * (-dist * dist / (2.0 * radius * radius * 0.25)).exp();
                    let idx = pixel_idx(cx + dx, cy + dy);
                    resource_data[idx] = (resource_data[idx] + boost).min(1.0);
                }
            }
        }

        let num_deserts = ((cluster_count_hint * 0.10).round() as i32).max(2);
        let desert_strength = (0.25 + params.resource_consumption * 2.5).clamp(0.2, 0.75);
        for _ in 0..num_deserts {
            let cx = rng.gen_range(0..w);
            let cy = rng.gen_range(0..h);
            let radius = rng.gen_range(25.0..50.0) * cluster_size_scale;
            let ir = radius as i32 + 1;
            for dy in -ir..=ir {
                for dx in -ir..=ir {
                    let dist = ((dx * dx + dy * dy) as f32).sqrt();
                    if dist > radius { continue; }
                    let reduction = desert_strength * (-dist * dist / (2.0 * radius * radius * 0.25)).exp();
                    let idx = pixel_idx(cx + dx, cy + dy);
                    resource_data[idx] = (resource_data[idx] - reduction).max(0.05);
                }
            }
        }

        let freq_x: f32 = rng.gen_range(1.0..4.0) * std::f32::consts::TAU / w as f32;
        let freq_y: f32 = rng.gen_range(1.0..4.0) * std::f32::consts::TAU / h as f32;
        let phase: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
        for py in 0..WORLD_HEIGHT {
            for px in 0..WORLD_WIDTH {
                let idx = (py * WORLD_WIDTH + px) as usize;
                let wave = (px as f32 * freq_x + py as f32 * freq_y + phase).sin() * 0.1;
                resource_data[idx] = (resource_data[idx] + wave).clamp(0.05, 1.0);
            }
        }

        let genome_a_flat: Vec<f32> = genome_a_data.iter().flat_map(|g| g.iter().copied()).collect();

        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;

        let create_f32_buffer = |label: &str, data: &[f32]| -> wgpu::Buffer {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage,
            })
        };

        let zeros_f32 = vec![0.0f32; n];
        let zeros_vec2 = vec![0.0f32; n * 2];
        let zeros_vec4 = vec![0.0f32; n * 4];

        let mass = [
            create_f32_buffer("mass_0", &mass_data),
            create_f32_buffer("mass_1", &zeros_f32),
        ];
        let energy = [
            create_f32_buffer("energy_0", &energy_data),
            create_f32_buffer("energy_1", &zeros_f32),
        ];
        let genome_a = [
            create_f32_buffer("genome_a_0", &genome_a_flat),
            create_f32_buffer("genome_a_1", &zeros_vec4),
        ];
        let genome_b = [
            create_f32_buffer("genome_b_0", &genome_b_data),
            create_f32_buffer("genome_b_1", &zeros_f32),
        ];

        let resource_map = create_f32_buffer("resource_map", &resource_data);
        let velocity = create_f32_buffer("velocity", &zeros_vec2);

        let mass_sum = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mass_sum"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sim_params = SimParams {
            width: WORLD_WIDTH, height: WORLD_HEIGHT, frame: 0, dt: DT,
            mutation_rate_mult: 1.0, predation_factor: 1.0,
            radius_cost_exp: 1.5, agg_mobility: 0.3, starvation_severity: 0.05,
            _pad1: 0, _pad2: 0, _pad3: 0,
        };
        let sim_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sim_params"),
            contents: bytemuck::bytes_of(&sim_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let velocity_params = VelocityParams { width: WORLD_WIDTH, height: WORLD_HEIGHT, frame: 0, _pad: 0 };
        let velocity_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("velocity_params"),
            contents: bytemuck::bytes_of(&velocity_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let resource_params = ResourceParams {
            width: WORLD_WIDTH, height: WORLD_HEIGHT,
            diffusion: 0.08, feed_rate: 0.010, consumption: 0.08,
            _pad1: 0, _pad2: 0, _pad3: 0,
        };
        let resource_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("resource_params"),
            contents: bytemuck::bytes_of(&resource_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let normalize_params = NormalizeParams {
            width: WORLD_WIDTH, height: WORLD_HEIGHT,
            target_mass_x1000: (target_total_mass() * 1000.0) as u32,
            damping_x1000: 300, enabled: 1, dust_floor_x1000: 2,
            _pad2: 0, _pad3: 0,
        };
        let normalize_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("normalize_params"),
            contents: bytemuck::bytes_of(&normalize_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let render_params = RenderParams {
            width: WORLD_WIDTH, height: WORLD_HEIGHT,
            visualization_mode: 0, show_legend: 0, time: 0.0, _pad: 0.0,
        };
        let render_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("render_params"),
            contents: bytemuck::bytes_of(&render_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let staging_usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let n_bytes_f32 = (n * std::mem::size_of::<f32>()) as u64;

        let staging_mass = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_mass"), size: n_bytes_f32, usage: staging_usage, mapped_at_creation: false,
        });
        let staging_energy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_energy"), size: n_bytes_f32, usage: staging_usage, mapped_at_creation: false,
        });
        let staging_genome_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_genome_a"), size: n_bytes_f32 * 4, usage: staging_usage, mapped_at_creation: false,
        });
        let staging_genome_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_genome_b"), size: n_bytes_f32, usage: staging_usage, mapped_at_creation: false,
        });
        let staging_resource = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_resource"), size: n_bytes_f32, usage: staging_usage, mapped_at_creation: false,
        });

        WorldState {
            current: 0, mass, energy, genome_a, genome_b,
            resource_map, velocity, mass_sum,
            staging_mass, staging_energy, staging_genome_a, staging_genome_b, staging_resource,
            sim_params_buffer, velocity_params_buffer, resource_params_buffer,
            normalize_params_buffer, render_params_buffer,
            frame: 0,
        }
    }
}
