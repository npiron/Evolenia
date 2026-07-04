// ============================================================================
// app/render.rs — Frame rendering, simulation encoding, and egui overlay
// ============================================================================

use std::time::Instant;

use crate::lab_ui;
use crate::metrics::SimDiagnostics;
use crate::pipeline::{create_pipelines, Pipelines};
use crate::renderer::HudPrepareConfig;
use crate::state_io;
use crate::world::*;

use super::AppState;

// ======================== Frame Rendering ========================

pub fn redraw(state: &mut AppState) {
    let win_w = state.surface_config.width;
    let win_h = state.surface_config.height;

    // FPS (exponential moving average)
    let now = Instant::now();
    let dt = now.duration_since(state.last_redraw).as_secs_f32().max(0.0001);
    state.last_redraw = now;
    state.fps = state.fps * 0.95 + (1.0 / dt) * 0.05;

    // Camera movement from held keys
    state.camera.apply_pan(state.keys.w, state.keys.s, state.keys.a, state.keys.d);
    state.camera.apply_zoom_keys(state.keys.e, state.keys.q);

    // Upload camera uniform
    state.queue.write_buffer(
        &state.pipelines.camera_buffer,
        0,
        bytemuck::bytes_of(&state.camera.uniforms(win_w, win_h)),
    );

    // Upload render params
    let render_params = RenderParams {
        width: WORLD_WIDTH, height: WORLD_HEIGHT,
        visualization_mode: state.sim_params.visualization_mode,
        show_legend: if state.lab.show_legend { 1 } else { 0 },
        time: state.world.frame as f32 * state.sim_params.time_step,
        _pad: 0.0,
    };
    state.queue.write_buffer(
        &state.world.render_params_buffer,
        0,
        bytemuck::bytes_of(&render_params),
    );

    // ---- egui frame ----
    let raw_input = state.egui_winit_state.take_egui_input(&state.window);
    let full_output = state.egui_ctx.run(raw_input, |ctx| {
        lab_ui::render_lab_ui(ctx, &mut state.sim_params, &mut state.lab);
    });
    state.egui_winit_state.handle_platform_output(&state.window, full_output.platform_output);

    // ---- Handle lab actions ----
    if state.lab.restart_requested {
        let seed = state.sim_params.effective_seed();
        state.world = WorldState::new_with_config(&state.device, seed, &state.sim_params);
        state.pipelines = create_pipelines(&state.device, &state.world, state.surface_config.format);
        state.lab.restart_requested = false;
        state.last_diag = None;
        state.lab.metrics_history.clear();
        state.lab.events.clear();
        state.lab.log_event(state.world.frame, "RESTART", "Simulation restarted");
        if let Some(s) = seed {
            state.lab.log_event(state.world.frame, "SEED", &format!("Seed: {}", s));
        }
        log::info!("Simulation restarted (seed: {:?})", seed);
    }

    // ---- Handle perturbation ----
    if state.sim_params.perturbation_active {
        state.world.apply_perturbation(&state.device, &state.queue, &state.sim_params);
        state.sim_params.perturbation_active = false;
        log::info!(
            "Perturbation applied: {} intensity={:.2} radius={:.2}",
            state.sim_params.perturbation_type.name(),
            state.sim_params.perturbation_intensity,
            state.sim_params.perturbation_radius,
        );
        state.lab.set_status(format!("Perturbation '{}' applied", state.sim_params.perturbation_type.name()));
    }

    state.diag_interval = state.lab.metrics_sample_interval.max(1);

    // ---- Prepare HUD (when Lab UI hidden and HUD enabled) ----
    if !state.lab.show_lab_ui && state.lab.hud_mode > 0 {
        let last_record = state.lab.metrics_history.last();
        state.hud.prepare(&HudPrepareConfig {
            device: &state.device,
            queue: &state.queue,
            params: &state.sim_params,
            frame: state.world.frame,
            fps: state.fps,
            win_w, win_h,
            hud_mode: state.lab.hud_mode,
            nes_species: last_record.map_or(0, |m| m.species),
            nes_total_mass: last_record.map_or(0.0, |m| m.total_mass),
            nes_entropy: last_record.map_or(0.0, |m| m.entropy),
            nes_avg_energy: last_record.map_or(0.0, |m| m.avg_energy),
            nes_live_fraction: last_record.map_or(0.0, |m| m.live_fraction),
            nes_predator_fraction: last_record.map_or(0.0, |m| m.predator_fraction),
            nes_prey_fraction: last_record.map_or(0.0, |m| m.prey_fraction),
        });
    }

    let dispatch_x = WORLD_WIDTH.div_ceil(WORKGROUP_X);
    let dispatch_y = WORLD_HEIGHT.div_ceil(WORKGROUP_Y);
    let dispatch_linear = total_pixels().div_ceil(256);

    // ---- Simulation steps ----
    if !state.sim_params.paused {
        let steps = state.sim_params.simulation_speed;
        for _ in 0..steps {
            state.world.update_step_uniforms_dynamic(&state.queue, &state.sim_params);
            let cur = state.world.cur();
            let mut sim_encoder = state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim_encoder"),
            });
            encode_simulation_passes(
                &mut sim_encoder, &state.pipelines, cur,
                dispatch_x, dispatch_y, dispatch_linear,
            );
            state.queue.submit(std::iter::once(sim_encoder.finish()));
            state.world.swap();
        }
    } else if state.lab.step_requested {
        state.world.update_step_uniforms_dynamic(&state.queue, &state.sim_params);
        let cur = state.world.cur();
        let mut sim_encoder = state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("step_encoder"),
        });
        encode_simulation_passes(
            &mut sim_encoder, &state.pipelines, cur,
            dispatch_x, dispatch_y, dispatch_linear,
        );
        state.queue.submit(std::iter::once(sim_encoder.finish()));
        state.world.swap();
        state.lab.step_requested = false;
        state.lab.log_event(state.world.frame, "CONTROL", "Single step");
    }

    // ---- Render pass ----
    let render_cur = 1 - state.world.cur();
    let mut encoder = state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("render_encoder"),
    });

    let output = match state.surface.get_current_texture() {
        Ok(t) => t,
        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
            state.surface.configure(&state.device, &state.surface_config);
            return;
        }
        Err(e) => {
            log::error!("Surface error: {:?}", e);
            return;
        }
    };

    let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
    let msaa_view = state.msaa_view.as_ref().expect("MSAA view not initialized");

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: msaa_view,
                resolve_target: Some(&view),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.015, g: 0.015, b: 0.04, a: 1.0 }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&state.pipelines.render_pipeline);
        pass.set_bind_group(0, &state.pipelines.render_bind_groups[render_cur], &[]);
        pass.draw(0..6, 0..1);
    }

    // HUD overlay
    if !state.lab.show_lab_ui && state.lab.hud_mode > 0 {
        let mut hud_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("hud_render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        state.hud.render(&mut hud_pass);
    }

    // ---- Screenshot capture ----
    let do_screenshot = state.lab.screenshot_requested;
    let mut screenshot_staging: Option<wgpu::Buffer> = None;
    let mut screenshot_padded_bpr: u32 = 0;

    if do_screenshot {
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let unpadded_bpr = win_w * 4;
        let padded_bpr = unpadded_bpr.div_ceil(align) * align;
        screenshot_padded_bpr = padded_bpr;

        let staging = state.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screenshot_staging"),
            size: (padded_bpr * win_h) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &output.texture, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(win_h),
                },
            },
            wgpu::Extent3d { width: win_w, height: win_h, depth_or_array_layers: 1 },
        );
        screenshot_staging = Some(staging);
    }

    state.queue.submit(std::iter::once(encoder.finish()));

    // ---- egui render pass ----
    let paint_jobs = state.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);

    for (id, image_delta) in &full_output.textures_delta.set {
        state.egui_renderer.update_texture(&state.device, &state.queue, *id, image_delta);
    }

    let screen_descriptor = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [win_w, win_h],
        pixels_per_point: full_output.pixels_per_point,
    };

    let mut egui_encoder = state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("egui_encoder"),
    });

    state.egui_renderer.update_buffers(
        &state.device, &state.queue, &mut egui_encoder, &paint_jobs, &screen_descriptor,
    );

    render_egui_pass(&state.egui_renderer, &mut egui_encoder, &view, &paint_jobs, &screen_descriptor);
    state.queue.submit(std::iter::once(egui_encoder.finish()));

    // ---- Read back screenshot ----
    if do_screenshot {
        if let Some(staging) = &screenshot_staging {
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| { let _ = tx.send(result); });
            state.device.poll(wgpu::Maintain::Wait);

            if let Ok(Ok(())) = rx.recv() {
                let data = slice.get_mapped_range();
                let mut rgba = Vec::with_capacity((win_w * win_h * 4) as usize);
                for row in 0..win_h {
                    let start = (row * screenshot_padded_bpr) as usize;
                    let end = start + (win_w * 4) as usize;
                    let row_data = &data[start..end];
                    for chunk in row_data.chunks_exact(4) {
                        rgba.push(chunk[2]); // R
                        rgba.push(chunk[1]); // G
                        rgba.push(chunk[0]); // B
                        rgba.push(chunk[3]); // A
                    }
                }
                drop(data);
                staging.unmap();

                match state.lab.save_screenshot(
                    state.world.frame, win_w, win_h, &rgba, state.sim_params.visualization_mode,
                ) {
                    Ok(path) => {
                        state.lab.set_status(format!("Screenshot saved: {:?}", path));
                        state.lab.log_event(state.world.frame, "SCREENSHOT", &format!("Saved to {:?}", path));
                    }
                    Err(e) => {
                        state.lab.set_status(format!("Screenshot failed: {}", e));
                        log::error!("Screenshot failed: {}", e);
                    }
                }
            }
        }
        state.lab.screenshot_requested = false;
    }

    // ---- Snapshot (state save) ----
    if state.lab.snapshot_requested {
        if let Some(snap) = state.world.readback_snapshot(&state.device, &state.queue) {
            let path = state.lab.run_dir.join(format!("snapshot_frame{:06}.snap", state.world.frame));
            match state_io::save_snapshot(path.to_str().unwrap_or("snapshot.snap"), &snap) {
                Ok(()) => {
                    state.lab.set_status(format!("Snapshot saved: {:?}", path));
                    state.lab.log_event(state.world.frame, "SNAPSHOT", &format!("Saved to {:?}", path));
                }
                Err(e) => {
                    log::error!("Snapshot save failed: {}", e);
                    state.lab.set_status(format!("Snapshot failed: {}", e));
                }
            }
        }
        state.lab.snapshot_requested = false;
    }

    output.present();

    for id in &full_output.textures_delta.free {
        state.egui_renderer.free_texture(id);
    }
    state.hud.trim();

    // ---- Periodic diagnostics ----
    if !state.sim_params.paused
        && state.world.frame > 0
        && state.world.frame.is_multiple_of(state.diag_interval)
    {
        if let Some(snap) = state.world.readback_snapshot(&state.device, &state.queue) {
            let diag = SimDiagnostics::from_snapshot(&snap)
                .with_target_mass(target_total_mass() * state.sim_params.target_mass_multiplier);
            state.lab.record_metrics(&diag, state.world.frame, state.fps);
            diag.log(state.world.frame, target_total_mass(), state.last_diag.as_ref());
            state.last_diag = Some(diag);
        }
    }

    state.window.request_redraw();
}

// ======================== egui Render Helper ========================

fn render_egui_pass(
    renderer: &egui_wgpu::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    paint_jobs: &[egui::ClippedPrimitive],
    screen_descriptor: &egui_wgpu::ScreenDescriptor,
) {
    let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("egui_render_pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    let mut pass = pass.forget_lifetime();
    renderer.render(&mut pass, paint_jobs, screen_descriptor);
}

// ======================== Simulation Encoding ========================

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
            label: Some("velocity_pass"), timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.velocity_pipeline);
        pass.set_bind_group(0, &pipelines.velocity_bind_groups[cur], &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    // Pass 2: Evolution
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("evolution_pass"), timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.evolution_pipeline);
        pass.set_bind_group(0, &pipelines.evolution_bind_groups[cur], &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    // Pass 3: Resource dynamics
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("resources_pass"), timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.resources_pipeline);
        pass.set_bind_group(0, &pipelines.resources_bind_groups[cur], &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    // Pass 4a: Sum total mass
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sum_mass_pass"), timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.sum_mass_pipeline);
        pass.set_bind_group(0, &pipelines.normalize_bind_groups[cur], &[]);
        pass.dispatch_workgroups(dispatch_linear, 1, 1);
    }

    // Pass 4b: Normalize mass
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("normalize_pass"), timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.normalize_pipeline);
        pass.set_bind_group(0, &pipelines.normalize_bind_groups[cur], &[]);
        pass.dispatch_workgroups(dispatch_linear, 1, 1);
    }
}
