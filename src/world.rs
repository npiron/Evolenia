// ============================================================================
// world.rs — EvoLenia v2
// WorldState: manages all GPU buffers (ping-pong pairs) and provides
// initialization with random seed clusters for the simulation.
// ============================================================================

use bytemuck::{Pod, Zeroable};
use rand::Rng;
use wgpu::util::DeviceExt;

// ======================== Constants ========================

pub const WORLD_WIDTH: u32 = 1024;
pub const WORLD_HEIGHT: u32 = 1024;
pub const WORKGROUP_X: u32 = 16;
pub const WORKGROUP_Y: u32 = 16;
pub const DT: f32 = 0.1;
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
    pub _pad0: u32,
    pub _pad1: u32,
}

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
        let mut genome_a_data = vec![[0.0f32; 4]; n]; // [r, mu, sigma, agg]
        let mut genome_b_data = vec![0.01f32; n]; // default mutation rate
        let resource_data = vec![1.0f32; n]; // full nutrients everywhere

        // Seed 50+ independent clusters with random genomes
        let num_clusters = 60;
        for _ in 0..num_clusters {
            let cx = rng.gen_range(0..WORLD_WIDTH) as i32;
            let cy = rng.gen_range(0..WORLD_HEIGHT) as i32;
            let cluster_radius = rng.gen_range(5..15) as f32;

            // Random genome for this cluster (all organisms in a cluster share genes)
            let gene_r: f32 = rng.gen_range(2.0..16.0);
            let gene_mu: f32 = rng.gen_range(0.1..0.9);
            let gene_sigma: f32 = rng.gen_range(0.02..0.25);
            let gene_agg: f32 = rng.gen_range(0.0..1.0);
            let gene_mut: f32 = rng.gen_range(0.001..0.03);

            let ir = cluster_radius as i32 + 1;
            for dy in -ir..=ir {
                for dx in -ir..=ir {
                    let dist = ((dx * dx + dy * dy) as f32).sqrt();
                    if dist > cluster_radius {
                        continue;
                    }

                    // Toroidal coordinates
                    let px = ((cx + dx) % WORLD_WIDTH as i32 + WORLD_WIDTH as i32)
                        % WORLD_WIDTH as i32;
                    let py = ((cy + dy) % WORLD_HEIGHT as i32 + WORLD_HEIGHT as i32)
                        % WORLD_HEIGHT as i32;
                    let idx = (py as u32 * WORLD_WIDTH + px as u32) as usize;

                    // Gaussian falloff from cluster center
                    let falloff =
                        (-dist * dist / (2.0 * cluster_radius * cluster_radius * 0.25)).exp();

                    mass_data[idx] = (mass_data[idx] + falloff).min(1.0);
                    energy_data[idx] = 0.5;
                    genome_a_data[idx] = [gene_r, gene_mu, gene_sigma, gene_agg];
                    genome_b_data[idx] = gene_mut;
                }
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
            _pad0: 0,
            _pad1: 0,
        };
        let render_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("render_params"),
            contents: bytemuck::bytes_of(&render_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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
}
