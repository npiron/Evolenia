// ============================================================================
// simulation.rs — EvoLenia v2
// Shared simulation encoding used by both GUI (app/render.rs) and headless mode.
// ============================================================================

use crate::pipeline::Pipelines;

/// Encode all 5 compute passes for one simulation step into the given encoder.
/// Used by both GUI rendering and headless batch runner.
pub fn encode_simulation_passes(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &Pipelines,
    cur: usize,
    dispatch_x: u32,
    dispatch_y: u32,
    dispatch_linear: u32,
) {
    // Pass 1: Velocity field
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("velocity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.velocity_pipeline);
        pass.set_bind_group(0, &pipelines.velocity_bind_groups[cur], &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    // Pass 2: Evolution
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("evolution_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.evolution_pipeline);
        pass.set_bind_group(0, &pipelines.evolution_bind_groups[cur], &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    // Pass 3: Resource dynamics
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("resources_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.resources_pipeline);
        pass.set_bind_group(0, &pipelines.resources_bind_groups[cur], &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    // Pass 4a: Sum total mass
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sum_mass_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.sum_mass_pipeline);
        pass.set_bind_group(0, &pipelines.normalize_bind_groups[cur], &[]);
        pass.dispatch_workgroups(dispatch_linear, 1, 1);
    }

    // Pass 4b: Normalize mass
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
