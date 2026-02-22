// ============================================================================
// headless.rs — EvoLenia v2
// Headless simulation runner for fast long-horizon batches.
// ============================================================================

use crate::pipeline::{create_pipelines, Pipelines};
use crate::state_io;
use crate::world::{total_pixels, WORKGROUP_X, WORKGROUP_Y, WorldState, WORLD_HEIGHT, WORLD_WIDTH};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct HeadlessConfig {
    pub frames: u32,
    pub load_state_path: Option<String>,
    pub save_state_path: Option<String>,
    pub progress_interval: u32,
}

impl Default for HeadlessConfig {
    fn default() -> Self {
        Self {
            frames: 10_000,
            load_state_path: None,
            save_state_path: None,
            progress_interval: 5000,
        }
    }
}

pub fn run_headless(config: &HeadlessConfig) -> Result<(), String> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| String::from("Failed to get GPU adapter for headless mode"))?;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("evolenia_headless_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                max_storage_buffers_per_shader_stage: 12,
                max_storage_buffer_binding_size: 256 * 1024 * 1024,
                ..Default::default()
            },
            memory_hints: Default::default(),
        },
        None,
    ))
    .map_err(|e| format!("Failed to create headless device: {e}"))?;

    let mut world = WorldState::new(&device);
    if let Some(path) = &config.load_state_path {
        let snap = state_io::load_snapshot(path)
            .map_err(|e| format!("Failed to load state {}: {}", path, e))?;
        if !world.apply_snapshot(&queue, &snap) {
            return Err(format!("Loaded state {} has incompatible dimensions", path));
        }
    }

    let pipelines = create_pipelines(&device, &world, wgpu::TextureFormat::Rgba8Unorm);

    let dispatch_x = (WORLD_WIDTH + WORKGROUP_X - 1) / WORKGROUP_X;
    let dispatch_y = (WORLD_HEIGHT + WORKGROUP_Y - 1) / WORKGROUP_Y;
    let dispatch_linear = (total_pixels() + 255) / 256;

    log::info!(
        "Headless run started: {} frames on {}x{}",
        config.frames,
        WORLD_WIDTH,
        WORLD_HEIGHT
    );

    let started = Instant::now();
    let mut last_report = Instant::now();
    let mut last_report_frame = 0u32;

    for step in 0..config.frames {
        world.update_step_uniforms(&queue);
        let cur = world.cur();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("headless_sim_encoder"),
        });
        encode_simulation_passes(
            &mut encoder,
            &pipelines,
            cur,
            dispatch_x,
            dispatch_y,
            dispatch_linear,
        );
        queue.submit(std::iter::once(encoder.finish()));
        world.swap();

        if config.progress_interval > 0 && (step + 1) % config.progress_interval == 0 {
            let done = step + 1;
            let total_elapsed = started.elapsed().as_secs_f64().max(1e-6);
            let total_fps = done as f64 / total_elapsed;

            let window_elapsed = last_report.elapsed().as_secs_f64().max(1e-6);
            let window_frames = done - last_report_frame;
            let window_fps = window_frames as f64 / window_elapsed;

            let remaining = config.frames.saturating_sub(done);
            let eta_secs = if total_fps > 1e-6 {
                remaining as f64 / total_fps
            } else {
                0.0
            };
            let eta_min = eta_secs / 60.0;

            log::info!(
                "Headless progress: {}/{} | fps={:.0} (window {:.0}) | ETA={:.1} min",
                done,
                config.frames,
                total_fps,
                window_fps,
                eta_min,
            );

            last_report = Instant::now();
            last_report_frame = done;
        }
    }

    if let Some(path) = &config.save_state_path {
        let snapshot = world
            .readback_snapshot(&device, &queue)
            .ok_or_else(|| String::from("GPU readback failed at end of headless run"))?;
        state_io::save_snapshot(path, &snapshot)
            .map_err(|e| format!("Failed to save snapshot {}: {}", path, e))?;
        log::info!("Saved final state to {}", path);
    }

    Ok(())
}

fn encode_simulation_passes(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &Pipelines,
    cur: usize,
    dispatch_x: u32,
    dispatch_y: u32,
    dispatch_linear: u32,
) {
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("velocity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.velocity_pipeline);
        pass.set_bind_group(0, &pipelines.velocity_bind_groups[cur], &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("evolution_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.evolution_pipeline);
        pass.set_bind_group(0, &pipelines.evolution_bind_groups[cur], &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("resources_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.resources_pipeline);
        pass.set_bind_group(0, &pipelines.resources_bind_groups[cur], &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sum_mass_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.sum_mass_pipeline);
        pass.set_bind_group(0, &pipelines.normalize_bind_groups[cur], &[]);
        pass.dispatch_workgroups(dispatch_linear, 1, 1);
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("normalize_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.normalize_pipeline);
        pass.set_bind_group(0, &pipelines.normalize_bind_groups[cur], &[]);
        pass.dispatch_workgroups(dispatch_linear, 1, 1);
    }
}
