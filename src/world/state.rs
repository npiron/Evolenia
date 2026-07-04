// ============================================================================
// world/state.rs — WorldState runtime methods (snapshot, swap, uniforms, perturbations, readback)
// ============================================================================

use bytemuck;

use super::constants::*;
use super::init::WorldState;
use super::types::*;
use crate::config::SimulationParams;

impl WorldState {
    /// Overwrite simulation buffers from a CPU snapshot.
    pub fn apply_snapshot(&mut self, queue: &wgpu::Queue, snapshot: &BufferSnapshot) -> bool {
        let n = total_pixels() as usize;
        if snapshot.mass.len() != n
            || snapshot.energy.len() != n
            || snapshot.genome_a.len() != n * 4
            || snapshot.genome_b.len() != n
            || snapshot.resource.len() != n
        {
            return false;
        }

        let write_mass = bytemuck::cast_slice(snapshot.mass.as_slice());
        let write_energy = bytemuck::cast_slice(snapshot.energy.as_slice());
        let write_genome_a = bytemuck::cast_slice(snapshot.genome_a.as_slice());
        let write_genome_b = bytemuck::cast_slice(snapshot.genome_b.as_slice());
        let write_resource = bytemuck::cast_slice(snapshot.resource.as_slice());

        for i in 0..2 {
            queue.write_buffer(&self.mass[i], 0, write_mass);
            queue.write_buffer(&self.energy[i], 0, write_energy);
            queue.write_buffer(&self.genome_a[i], 0, write_genome_a);
            queue.write_buffer(&self.genome_b[i], 0, write_genome_b);
        }
        queue.write_buffer(&self.resource_map, 0, write_resource);

        self.current = 0;
        true
    }

    /// Swap ping-pong buffers after a frame.
    pub fn swap(&mut self) {
        self.current = 1 - self.current;
        self.frame += 1;
    }

    /// Index of the current (read) buffer.
    pub fn cur(&self) -> usize {
        self.current
    }

    /// Index of the next (write) buffer.
    #[allow(dead_code)]
    pub fn next(&self) -> usize {
        1 - self.current
    }

    /// Update per-step uniforms (static defaults).
    pub fn update_step_uniforms(&self, queue: &wgpu::Queue) {
        let sim_params = SimParams {
            width: WORLD_WIDTH, height: WORLD_HEIGHT, frame: self.frame, dt: DT,
            mutation_rate_mult: 1.0, predation_factor: 1.0,
            radius_cost_exp: 1.5, agg_mobility: 0.3, starvation_severity: 0.05,
            _pad1: 0, _pad2: 0, _pad3: 0,
        };
        queue.write_buffer(&self.sim_params_buffer, 0, bytemuck::bytes_of(&sim_params));
        queue.write_buffer(&self.mass_sum, 0, bytemuck::bytes_of(&[0u32; 2]));
    }

    /// Update all uniforms using dynamic parameters from the Research Lab UI.
    pub fn update_step_uniforms_dynamic(&self, queue: &wgpu::Queue, params: &SimulationParams) {
        let sim_params = SimParams {
            width: WORLD_WIDTH,
            height: WORLD_HEIGHT,
            frame: self.frame,
            dt: DT * params.time_step,
            mutation_rate_mult: params.mutation_rate,
            predation_factor: params.predation_factor,
            radius_cost_exp: params.radius_cost_exponent,
            agg_mobility: params.agg_mobility_tradeoff,
            starvation_severity: params.starvation_severity,
            _pad1: 0, _pad2: 0, _pad3: 0,
        };
        queue.write_buffer(&self.sim_params_buffer, 0, bytemuck::bytes_of(&sim_params));

        let resource_params = ResourceParams {
            width: WORLD_WIDTH, height: WORLD_HEIGHT,
            diffusion: params.resource_diffusion,
            feed_rate: params.resource_feed_rate,
            consumption: params.resource_consumption,
            _pad1: 0, _pad2: 0, _pad3: 0,
        };
        queue.write_buffer(&self.resource_params_buffer, 0, bytemuck::bytes_of(&resource_params));

        let normalize_params = NormalizeParams {
            width: WORLD_WIDTH, height: WORLD_HEIGHT,
            target_mass_x1000: (target_total_mass() * params.target_mass_multiplier * 1000.0) as u32,
            damping_x1000: (params.mass_damping * 1000.0) as u32,
            enabled: if params.mass_normalization_enabled { 1 } else { 0 },
            dust_floor_x1000: 2,
            _pad2: 0, _pad3: 0,
        };
        queue.write_buffer(&self.normalize_params_buffer, 0, bytemuck::bytes_of(&normalize_params));

        queue.write_buffer(&self.mass_sum, 0, bytemuck::bytes_of(&[0u32; 2]));
    }

    /// Apply an ecological perturbation to the simulation buffers.
    pub fn apply_perturbation(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        params: &SimulationParams,
    ) {
        use crate::config::PerturbationType;

        let snap = match self.readback_snapshot(device, queue) {
            Some(s) => s,
            None => return,
        };

        let w = WORLD_WIDTH as f32;
        let h = WORLD_HEIGHT as f32;
        let cx = params.perturbation_center_x * w;
        let cy = params.perturbation_center_y * h;
        let radius = params.perturbation_radius * w;
        let intensity = params.perturbation_intensity;

        let mut resource = snap.resource.clone();
        let mut mass = snap.mass.clone();
        let mut energy = snap.energy.clone();
        let mut genome_b = snap.genome_b.clone();

        let cur = self.cur();

        for py in 0..WORLD_HEIGHT {
            for px in 0..WORLD_WIDTH {
                let idx = (py * WORLD_WIDTH + px) as usize;
                let mut dx = px as f32 - cx;
                let mut dy = py as f32 - cy;
                if dx > w * 0.5 { dx -= w; }
                if dx < -w * 0.5 { dx += w; }
                if dy > h * 0.5 { dy -= h; }
                if dy < -h * 0.5 { dy += h; }
                let dist = (dx * dx + dy * dy).sqrt();
                if dist > radius { continue; }

                let falloff = 1.0 - dist / radius;
                match params.perturbation_type {
                    PerturbationType::Drought => {
                        resource[idx] *= 1.0 - intensity * falloff * 0.8;
                        resource[idx] = resource[idx].max(0.01);
                    }
                    PerturbationType::NutrientPulse => {
                        resource[idx] += intensity * falloff * 0.5;
                        resource[idx] = resource[idx].min(1.0);
                    }
                    PerturbationType::MassStorm => {
                        let kill = intensity * falloff * 0.7;
                        mass[idx] *= 1.0 - kill;
                        energy[idx] *= 1.0 - kill * 0.5;
                    }
                    PerturbationType::MutationBurst => {
                        if mass[idx] > 0.01 {
                            genome_b[idx] = (genome_b[idx] + intensity * falloff * 0.005).min(0.01);
                        }
                    }
                    PerturbationType::None => {}
                }
            }
        }

        queue.write_buffer(&self.resource_map, 0, bytemuck::cast_slice(&resource));
        queue.write_buffer(&self.mass[cur], 0, bytemuck::cast_slice(&mass));
        queue.write_buffer(&self.energy[cur], 0, bytemuck::cast_slice(&energy));
        queue.write_buffer(&self.genome_b[cur], 0, bytemuck::cast_slice(&genome_b));

        log::info!(
            "Perturbation applied: {:?} at ({:.0},{:.0}) r={:.0} i={:.2}",
            params.perturbation_type, cx, cy, radius, intensity
        );
    }

    /// Perform a synchronous GPU readback of all simulation buffers.
    pub fn readback_snapshot(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Option<BufferSnapshot> {
        let n = total_pixels() as usize;
        let n_bytes = (n * std::mem::size_of::<f32>()) as u64;
        let cur = self.cur();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.mass[cur], 0, &self.staging_mass, 0, n_bytes);
        encoder.copy_buffer_to_buffer(&self.energy[cur], 0, &self.staging_energy, 0, n_bytes);
        encoder.copy_buffer_to_buffer(&self.genome_a[cur], 0, &self.staging_genome_a, 0, n_bytes * 4);
        encoder.copy_buffer_to_buffer(&self.genome_b[cur], 0, &self.staging_genome_b, 0, n_bytes);
        encoder.copy_buffer_to_buffer(&self.resource_map, 0, &self.staging_resource, 0, n_bytes);
        queue.submit(std::iter::once(encoder.finish()));

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
