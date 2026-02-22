// ============================================================================
// world.rs — EvoLenia v2
// WorldState: manages all GPU buffers (ping-pong pairs) and provides
// initialization with random seed clusters for the simulation.
// ============================================================================

use bytemuck::{Pod, Zeroable};
use rand::Rng;
use wgpu::util::DeviceExt;

// ======================== Constants ========================

// Performance tuning:
// - 512×512 = 4× faster than 1024×1024 (good for development/testing)
// - 1024×1024 = balanced (default, ~60 FPS on M1 Pro)
// - 2048×2048 = highest quality (requires powerful GPU)
pub const WORLD_WIDTH: u32 = 512;   // Try 512 for 4× speed boost
pub const WORLD_HEIGHT: u32 = 512;
pub const WORKGROUP_X: u32 = 16;
pub const WORKGROUP_Y: u32 = 16;
pub const DT: f32 = 0.1;        // reduced for stability (was 0.1), try 0.1 for 2× speed
pub const TARGET_FILL: f32 = 0.15; // 15% initial mass fill

pub fn total_pixels() -> u32 {
    WORLD_WIDTH * WORLD_HEIGHT
}

pub fn target_total_mass() -> f32 {
    WORLD_WIDTH as f32 * WORLD_HEIGHT as f32 * TARGET_FILL
}

// ======================== Uniform Structs ========================

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SimParams {
    pub width: u32,
    pub height: u32,
    pub frame: u32,
    pub dt: f32,
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
    pub frame: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct NormalizeParams {
    pub width: u32,
    pub height: u32,
    pub target_mass_x1000: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct RenderParams {
    pub width: u32,
    pub height: u32,
    pub visualization_mode: u32,
    pub _pad: u32,
}

// ======================== WorldState ========================

/// Raw CPU-side snapshot of simulation buffers (obtained via GPU readback).
pub struct BufferSnapshot {
    pub mass: Vec<f32>,
    pub energy: Vec<f32>,
    pub genome_a: Vec<f32>, // flat vec4 per pixel (len = n*4)
    pub genome_b: Vec<f32>,
    pub resource: Vec<f32>,
}

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
    pub fn new(device: &wgpu::Device) -> Self {
        let n = total_pixels() as usize;
        let mut rng = rand::thread_rng();

        // ---- Initialize data on CPU ----
        let mut mass_data = vec![0.0f32; n];
        let mut energy_data = vec![0.5f32; n]; // uniform initial energy
        // CRITICAL: default genome must have valid values even for empty pixels.
        // sigma=0 causes division by zero in the growth function (exp(-x²/2σ²)).
        // Using safe defaults: r=5, mu=0.5, sigma=0.15, agg=0
        let mut genome_a_data = vec![[5.0f32, 0.5, 0.15, 0.0]; n]; // [r, mu, sigma, agg]
        let mut genome_b_data = vec![0.01f32; n]; // default mutation rate
        let mut resource_data = vec![1.0f32; n]; // full nutrients everywhere

        // ======================== Seed Patterns ========================
        // Five distinct pattern types to create diverse initial ecosystems:
        //   1. Gaussian clusters — classic circular colonies
        //   2. Rings / annuli — hollow donut-shaped organisms
        //   3. Lines / filaments — elongated wall-like structures
        //   4. Spirals — rotating arm patterns
        //   5. Scattered noise patches — diffuse low-density clouds

        let w = WORLD_WIDTH as i32;
        let h = WORLD_HEIGHT as i32;

        // Helper: toroidal pixel index
        let pixel_idx = |px: i32, py: i32| -> usize {
            let wx = ((px % w) + w) % w;
            let wy = ((py % h) + h) % h;
            (wy as u32 * WORLD_WIDTH + wx as u32) as usize
        };

        // Helper: write a pixel with mass/genome, blending with existing mass
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

        // --- Random genome generator ---
        let random_genome = |rng: &mut rand::rngs::ThreadRng| -> ([f32; 4], f32) {
            let gene_r: f32 = rng.gen_range(3.0..9.0);
            let gene_mu: f32 = rng.gen_range(0.12..0.30);
            let gene_sigma: f32 = rng.gen_range(0.04..0.18);
            let gene_agg: f32 = rng.gen_range(0.0..0.6);
            let gene_mut: f32 = rng.gen_range(0.0005..0.008);
            ([gene_r, gene_mu, gene_sigma, gene_agg], gene_mut)
        };

        // ---- PATTERN 1: Gaussian clusters (classic) ----
        let num_clusters = 30;
        for _ in 0..num_clusters {
            let cx = rng.gen_range(0..w);
            let cy = rng.gen_range(0..h);
            let radius = rng.gen_range(5..15) as f32;
            let (genome, mut_rate) = random_genome(&mut rng);

            let ir = radius as i32 + 1;
            for dy in -ir..=ir {
                for dx in -ir..=ir {
                    let dist = ((dx * dx + dy * dy) as f32).sqrt();
                    if dist > radius { continue; }
                    let falloff = (-dist * dist / (2.0 * radius * radius * 0.25)).exp();
                    let idx = pixel_idx(cx + dx, cy + dy);
                    stamp(&mut mass_data, &mut energy_data, &mut genome_a_data, &mut genome_b_data,
                          idx, falloff, 0.5, genome, mut_rate);
                }
            }
        }

        // ---- PATTERN 2: Rings / annuli ----
        let num_rings = 8;
        for _ in 0..num_rings {
            let cx = rng.gen_range(0..w);
            let cy = rng.gen_range(0..h);
            let outer_r = rng.gen_range(10..25) as f32;
            let inner_r = outer_r * rng.gen_range(0.4..0.7);
            let thickness = (outer_r - inner_r).max(2.0);
            let (genome, mut_rate) = random_genome(&mut rng);

            let ir = outer_r as i32 + 1;
            for dy in -ir..=ir {
                for dx in -ir..=ir {
                    let dist = ((dx * dx + dy * dy) as f32).sqrt();
                    if dist > outer_r || dist < inner_r { continue; }
                    // Smooth falloff at both edges
                    let edge_outer = 1.0 - ((dist - outer_r + thickness * 0.3) / (thickness * 0.3)).max(0.0);
                    let edge_inner = ((dist - inner_r) / (thickness * 0.3)).min(1.0);
                    let m = (edge_outer * edge_inner).clamp(0.0, 1.0);
                    if m < 0.01 { continue; }
                    let idx = pixel_idx(cx + dx, cy + dy);
                    stamp(&mut mass_data, &mut energy_data, &mut genome_a_data, &mut genome_b_data,
                          idx, m * 0.8, 0.6, genome, mut_rate);
                }
            }
        }

        // ---- PATTERN 3: Lines / filaments ----
        let num_lines = 6;
        for _ in 0..num_lines {
            let x0 = rng.gen_range(0..w);
            let y0 = rng.gen_range(0..h);
            let angle: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
            let length = rng.gen_range(30..80) as f32;
            let half_width = rng.gen_range(1.5..4.0_f32);
            let (genome, mut_rate) = random_genome(&mut rng);
            // Line with slight curve
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
                        stamp(&mut mass_data, &mut energy_data, &mut genome_a_data, &mut genome_b_data,
                              idx, m * 0.7, 0.5, genome, mut_rate);
                    }
                }
            }
        }

        // ---- PATTERN 4: Spirals ----
        let num_spirals = 4;
        for _ in 0..num_spirals {
            let cx = rng.gen_range(0..w) as f32;
            let cy = rng.gen_range(0..h) as f32;
            let arms: u32 = rng.gen_range(2..5);
            let max_angle: f32 = rng.gen_range(3.0..6.0); // radians of spiral
            let scale = rng.gen_range(15.0..35.0_f32);
            let arm_width = rng.gen_range(1.5..3.5_f32);
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
                            let m = (1.0 - d / arm_width) * (1.0 - t * 0.3); // fade at tip
                            if m < 0.01 { continue; }
                            let idx = pixel_idx(sx as i32 + dx, sy as i32 + dy);
                            stamp(&mut mass_data, &mut energy_data, &mut genome_a_data, &mut genome_b_data,
                                  idx, m * 0.6, 0.55, genome, mut_rate);
                        }
                    }
                }
            }
        }

        // ---- PATTERN 5: Scattered noise patches (diffuse clouds) ----
        let num_patches = 10;
        for _ in 0..num_patches {
            let cx = rng.gen_range(0..w);
            let cy = rng.gen_range(0..h);
            let patch_r = rng.gen_range(15..40) as i32;
            let density: f32 = rng.gen_range(0.05..0.15);
            let (genome, mut_rate) = random_genome(&mut rng);

            for dy in -patch_r..=patch_r {
                for dx in -patch_r..=patch_r {
                    let dist = ((dx * dx + dy * dy) as f32).sqrt();
                    if dist > patch_r as f32 { continue; }
                    // Random sparse fill within patch
                    if rng.gen::<f32>() > density { continue; }
                    let falloff = 1.0 - dist / patch_r as f32;
                    let m = falloff * rng.gen_range(0.1..0.5);
                    let idx = pixel_idx(cx + dx, cy + dy);
                    stamp(&mut mass_data, &mut energy_data, &mut genome_a_data, &mut genome_b_data,
                          idx, m, 0.4, genome, mut_rate);
                }
            }
        }

        // ---- PATTERN 6: Apex predator nests (high aggressivity, small, high energy) ----
        let num_predators = 5;
        for _ in 0..num_predators {
            let cx = rng.gen_range(0..w);
            let cy = rng.gen_range(0..h);
            let radius = rng.gen_range(3..7) as f32;
            let gene_r: f32 = rng.gen_range(4.0..7.0);
            let gene_mu: f32 = rng.gen_range(0.15..0.25);
            let gene_sigma: f32 = rng.gen_range(0.06..0.12);
            let gene_agg: f32 = rng.gen_range(0.7..1.0); // high aggressivity
            let gene_mut: f32 = rng.gen_range(0.001..0.005);
            let genome = [gene_r, gene_mu, gene_sigma, gene_agg];

            let ir = radius as i32 + 1;
            for dy in -ir..=ir {
                for dx in -ir..=ir {
                    let dist = ((dx * dx + dy * dy) as f32).sqrt();
                    if dist > radius { continue; }
                    let m = (-dist * dist / (2.0 * radius * radius * 0.3)).exp();
                    let idx = pixel_idx(cx + dx, cy + dy);
                    stamp(&mut mass_data, &mut energy_data, &mut genome_a_data, &mut genome_b_data,
                          idx, m * 0.9, 0.8, genome, gene_mut);
                }
            }
        }

        // ======================== Resource Map Heterogeneity ========================
        // Instead of uniform nutrients, create a varied landscape:
        // - Fertile zones (nutrient-rich)
        // - Desert zones (nutrient-poor)
        // - Gradient bands

        // Base: slightly reduced uniform nutrients
        for r in resource_data.iter_mut() {
            *r = 0.7;
        }

        // Fertile oases (high nutrients)
        let num_oases = 12;
        for _ in 0..num_oases {
            let cx = rng.gen_range(0..w);
            let cy = rng.gen_range(0..h);
            let radius = rng.gen_range(20..60) as f32;
            let ir = radius as i32 + 1;
            for dy in -ir..=ir {
                for dx in -ir..=ir {
                    let dist = ((dx * dx + dy * dy) as f32).sqrt();
                    if dist > radius { continue; }
                    let boost = 0.3 * (-dist * dist / (2.0 * radius * radius * 0.25)).exp();
                    let idx = pixel_idx(cx + dx, cy + dy);
                    resource_data[idx] = (resource_data[idx] + boost).min(1.0);
                }
            }
        }

        // Desert zones (low nutrients)
        let num_deserts = 6;
        for _ in 0..num_deserts {
            let cx = rng.gen_range(0..w);
            let cy = rng.gen_range(0..h);
            let radius = rng.gen_range(25..50) as f32;
            let ir = radius as i32 + 1;
            for dy in -ir..=ir {
                for dx in -ir..=ir {
                    let dist = ((dx * dx + dy * dy) as f32).sqrt();
                    if dist > radius { continue; }
                    let reduction = 0.5 * (-dist * dist / (2.0 * radius * radius * 0.25)).exp();
                    let idx = pixel_idx(cx + dx, cy + dy);
                    resource_data[idx] = (resource_data[idx] - reduction).max(0.05);
                }
            }
        }

        // Sinusoidal gradient bands (creates corridors)
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

        // Flatten genome_a to f32 for bytemuck
        let genome_a_flat: Vec<f32> = genome_a_data.iter().flat_map(|g| g.iter().copied()).collect();

        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        // ---- Create GPU Buffers ----
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

        // Ping-pong pairs
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

        // Single buffers
        let resource_map = create_f32_buffer("resource_map", &resource_data);
        let velocity = create_f32_buffer("velocity", &zeros_vec2);

        // Atomic sum buffer for normalization (2 atomic u32s)
        let mass_sum = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mass_sum"),
            size: 8, // 2 x u32
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---- Uniform Buffers ----
        let sim_params = SimParams {
            width: WORLD_WIDTH,
            height: WORLD_HEIGHT,
            frame: 0,
            dt: DT,
        };
        let sim_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sim_params"),
            contents: bytemuck::bytes_of(&sim_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let velocity_params = VelocityParams {
            width: WORLD_WIDTH,
            height: WORLD_HEIGHT,
            frame: 0,
            _pad: 0,
        };
        let velocity_params_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("velocity_params"),
                contents: bytemuck::bytes_of(&velocity_params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let resource_params = ResourceParams {
            width: WORLD_WIDTH,
            height: WORLD_HEIGHT,
            frame: 0,
            _pad: 0,
        };
        let resource_params_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("resource_params"),
                contents: bytemuck::bytes_of(&resource_params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let normalize_params = NormalizeParams {
            width: WORLD_WIDTH,
            height: WORLD_HEIGHT,
            target_mass_x1000: (target_total_mass() * 1000.0) as u32,
            _pad: 0,
        };
        let normalize_params_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("normalize_params"),
                contents: bytemuck::bytes_of(&normalize_params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let render_params = RenderParams {
            width: WORLD_WIDTH,
            height: WORLD_HEIGHT,
            visualization_mode: 0, // Default: Species Color
            _pad: 0,
        };
        let render_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("render_params"),
            contents: bytemuck::bytes_of(&render_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // ---- Staging Buffers for CPU readback ----
        let staging_usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let n_bytes_f32 = (n * std::mem::size_of::<f32>()) as u64;

        let staging_mass = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_mass"),
            size: n_bytes_f32,
            usage: staging_usage,
            mapped_at_creation: false,
        });
        let staging_energy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_energy"),
            size: n_bytes_f32,
            usage: staging_usage,
            mapped_at_creation: false,
        });
        let staging_genome_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_genome_a"),
            size: n_bytes_f32 * 4, // vec4 per pixel
            usage: staging_usage,
            mapped_at_creation: false,
        });
        let staging_genome_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_genome_b"),
            size: n_bytes_f32,
            usage: staging_usage,
            mapped_at_creation: false,
        });
        let staging_resource = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_resource"),
            size: n_bytes_f32,
            usage: staging_usage,
            mapped_at_creation: false,
        });

        WorldState {
            current: 0,
            mass,
            energy,
            genome_a,
            genome_b,
            resource_map,
            velocity,
            mass_sum,
            staging_mass,
            staging_energy,
            staging_genome_a,
            staging_genome_b,
            staging_resource,
            sim_params_buffer,
            velocity_params_buffer,
            resource_params_buffer,
            normalize_params_buffer,
            render_params_buffer,
            frame: 0,
        }
    }

    /// Swap ping-pong buffers after a frame
    pub fn swap(&mut self) {
        self.current = 1 - self.current;
        self.frame += 1;
    }

    /// Index of the current (read) buffer
    pub fn cur(&self) -> usize {
        self.current
    }

    /// Index of the next (write) buffer
    #[allow(dead_code)]
    pub fn next(&self) -> usize {
        1 - self.current
    }

    /// Update the frame counter in uniform buffers
    pub fn update_uniforms(&self, queue: &wgpu::Queue) {
        let sim_params = SimParams {
            width: WORLD_WIDTH,
            height: WORLD_HEIGHT,
            frame: self.frame,
            dt: DT,
        };
        queue.write_buffer(&self.sim_params_buffer, 0, bytemuck::bytes_of(&sim_params));

        let velocity_params = VelocityParams {
            width: WORLD_WIDTH,
            height: WORLD_HEIGHT,
            frame: self.frame,
            _pad: 0,
        };
        queue.write_buffer(
            &self.velocity_params_buffer,
            0,
            bytemuck::bytes_of(&velocity_params),
        );

        let resource_params = ResourceParams {
            width: WORLD_WIDTH,
            height: WORLD_HEIGHT,
            frame: self.frame,
            _pad: 0,
        };
        queue.write_buffer(
            &self.resource_params_buffer,
            0,
            bytemuck::bytes_of(&resource_params),
        );

        // Reset mass_sum atomic to 0 before each normalization pass
        queue.write_buffer(&self.mass_sum, 0, bytemuck::bytes_of(&[0u32; 2]));
    }

    /// Perform a synchronous GPU readback of all simulation buffers.
    /// This is expensive — call only every N frames for diagnostics.
    pub fn readback_snapshot(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Option<BufferSnapshot> {
        let n = total_pixels() as usize;
        let n_bytes = (n * std::mem::size_of::<f32>()) as u64;
        let cur = self.cur();

        // Encode copy commands: GPU storage → staging
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.mass[cur], 0, &self.staging_mass, 0, n_bytes);
        encoder.copy_buffer_to_buffer(&self.energy[cur], 0, &self.staging_energy, 0, n_bytes);
        encoder.copy_buffer_to_buffer(&self.genome_a[cur], 0, &self.staging_genome_a, 0, n_bytes * 4);
        encoder.copy_buffer_to_buffer(&self.genome_b[cur], 0, &self.staging_genome_b, 0, n_bytes);
        encoder.copy_buffer_to_buffer(&self.resource_map, 0, &self.staging_resource, 0, n_bytes);
        queue.submit(std::iter::once(encoder.finish()));

        // Helper: map a staging buffer and extract f32 data
        let read_staging = |buf: &wgpu::Buffer, count: usize| -> Option<Vec<f32>> {
            let slice = buf.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            device.poll(wgpu::Maintain::Wait);
            rx.recv().ok()?.ok()?;
            let data = slice.get_mapped_range();
            let floats: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            buf.unmap();
            if floats.len() >= count { Some(floats) } else { None }
        };

        let mass = read_staging(&self.staging_mass, n)?;
        let energy = read_staging(&self.staging_energy, n)?;
        let genome_a = read_staging(&self.staging_genome_a, n * 4)?;
        let genome_b = read_staging(&self.staging_genome_b, n)?;
        let resource = read_staging(&self.staging_resource, n)?;

        Some(BufferSnapshot { mass, energy, genome_a, genome_b, resource })
    }
}
