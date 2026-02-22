// ============================================================================
// app.rs — EvoLenia v2 + Research Lab
// Application state and winit event-loop handler with egui UI integration.
// ============================================================================

use std::sync::Arc;
use std::time::Instant;

use winit::{
    application::ApplicationHandler,
    event::{MouseScrollDelta, WindowEvent},
    keyboard::{Key, NamedKey},
    window::{Window, WindowAttributes},
};

use crate::camera::CameraState;
use crate::config::{SimulationParams, VIS_MODE_COUNT};
use crate::input::KeysHeld;
use crate::lab::LabState;
use crate::lab_ui;
use crate::metrics::SimDiagnostics;
use crate::pipeline::{create_pipelines, Pipelines};
use crate::renderer::HudRenderer;
use crate::state_io;
use crate::world::*;

// ======================== Application ========================

pub struct App {
    state: Option<AppState>,
    config: AppConfig,
}

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub initial_state_path: Option<String>,
    pub diag_interval: u32,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            initial_state_path: None,
            diag_interval: 300,
        }
    }
}

struct AppState {
    // GPU
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,

    // Simulation
    world: WorldState,
    pipelines: Pipelines,

    // Window
    window: Arc<Window>,

    // Camera & Input
    camera: CameraState,
    keys: KeysHeld,
    sim_params: SimulationParams,

    // HUD (minimal, kept as fallback)
    hud: HudRenderer,

    // egui
    egui_ctx: egui::Context,
    egui_winit_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,

    // Research Lab
    lab: LabState,

    // Timing
    last_redraw: Instant,
    fps: f32,

    // Diagnostics
    last_diag: Option<SimDiagnostics>,
    diag_interval: u32,
}

impl App {
    pub fn new(config: AppConfig) -> Self {
        Self { state: None, config }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window_attrs = WindowAttributes::default()
            .with_title("EvoLenia v2 — Research Lab")
            .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 1024u32));

        let window = Arc::new(event_loop.create_window(window_attrs).unwrap());

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let (device, queue, surface_config) =
            pollster::block_on(init_gpu(&instance, &surface, &window));

        surface.configure(&device, &surface_config);

        let mut world = WorldState::new(&device);
        if let Some(path) = &self.config.initial_state_path {
            match state_io::load_snapshot(path) {
                Ok(snapshot) => {
                    if world.apply_snapshot(&queue, &snapshot) {
                        log::info!("Loaded simulation state from {}", path);
                    } else {
                        log::warn!("State file {} has incompatible dimensions; using fresh world", path);
                    }
                }
                Err(err) => {
                    log::warn!("Failed to load state from {}: {}", path, err);
                }
            }
        }
        let pipelines = create_pipelines(&device, &world, surface_config.format);
        let hud = HudRenderer::new(&device, &queue, surface_config.format);

        // ---- Initialize egui ----
        let egui_ctx = egui::Context::default();
        // Dark theme with slightly transparent backgrounds for overlay feel
        let mut visuals = egui::Visuals::dark();
        visuals.window_fill = egui::Color32::from_rgba_premultiplied(27, 27, 35, 235);
        visuals.panel_fill = egui::Color32::from_rgba_premultiplied(20, 20, 28, 230);
        egui_ctx.set_visuals(visuals);

        let egui_winit_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            event_loop,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            surface_config.format,
            None,
            1,
            false,
        );

        log::info!(
            "EvoLenia v2 Research Lab initialized: {}x{}, target mass = {:.0}",
            WORLD_WIDTH,
            WORLD_HEIGHT,
            target_total_mass()
        );

        self.state = Some(AppState {
            device,
            queue,
            surface,
            surface_config,
            world,
            pipelines,
            window: window.clone(),
            camera: CameraState::default(),
            keys: KeysHeld::default(),
            sim_params: SimulationParams::default(),
            hud,
            egui_ctx,
            egui_winit_state,
            egui_renderer,
            lab: LabState::default(),
            last_redraw: Instant::now(),
            fps: 0.0,
            last_diag: None,
            diag_interval: self.config.diag_interval.max(1),
        });

        // Initial redraw — required on macOS with winit 0.30
        window.request_redraw();
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else {
            return;
        };

        // Pass events to egui first
        let egui_response = state.egui_winit_state.on_window_event(&state.window, &event);

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput { event, .. } => {
                // Always handle global hotkeys (F1, F9, F12, Escape)
                // Other keys only if egui didn't consume them
                handle_keyboard(state, event_loop, &event, egui_response.consumed);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                if !egui_response.consumed {
                    let scroll = match &delta {
                        MouseScrollDelta::LineDelta(_, y) => *y,
                        MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                    };
                    state.camera.apply_scroll(scroll);
                }
            }

            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    state.surface_config.width = new_size.width;
                    state.surface_config.height = new_size.height;
                    state.surface.configure(&state.device, &state.surface_config);
                }
            }

            WindowEvent::RedrawRequested => {
                redraw(state);
            }

            _ => {}
        }
    }
}

// ======================== GPU Initialization ========================

async fn init_gpu(
    instance: &wgpu::Instance,
    surface: &wgpu::Surface<'_>,
    window: &Window,
) -> (wgpu::Device, wgpu::Queue, wgpu::SurfaceConfiguration) {
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(surface),
            force_fallback_adapter: false,
        })
        .await
        .expect(
            "Failed to find a suitable GPU adapter.\n\
             EvoLenia requires a GPU with Vulkan, Metal, or DX12 support.",
        );

    log::info!("GPU: {}", adapter.get_info().name);

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("evolenia_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 12,
                    max_storage_buffer_binding_size: 256 * 1024 * 1024,
                    ..Default::default()
                },
                memory_hints: Default::default(),
            },
            None,
        )
        .await
        .expect("Failed to create device");

    let size = window.inner_size();
    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps
        .formats
        .iter()
        .find(|f| f.is_srgb())
        .copied()
        .unwrap_or(surface_caps.formats[0]);

    // Use Mailbox (uncapped FPS, no tearing) if available, else Immediate, else Fifo.
    let present_mode = if surface_caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
        log::info!("Present mode: Mailbox (uncapped FPS)");
        wgpu::PresentMode::Mailbox
    } else if surface_caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
        log::info!("Present mode: Immediate (uncapped FPS)");
        wgpu::PresentMode::Immediate
    } else {
        log::info!("Present mode: Fifo (VSync ON)");
        wgpu::PresentMode::Fifo
    };

    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        format: surface_format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode,
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };

    (device, queue, surface_config)
}

// ======================== Keyboard Handling ========================

fn handle_keyboard(
    state: &mut AppState,
    event_loop: &winit::event_loop::ActiveEventLoop,
    event: &winit::event::KeyEvent,
    egui_consumed: bool,
) {
    let pressed = event.state.is_pressed();

    // Global hotkeys — always handled, even when egui has focus
    match &event.logical_key {
        Key::Named(NamedKey::Escape) if pressed => event_loop.exit(),
        Key::Named(NamedKey::F1) if pressed => {
            state.lab.show_lab_ui = !state.lab.show_lab_ui;
            log::info!("Lab UI: {}", if state.lab.show_lab_ui { "ON" } else { "OFF" });
        }
        Key::Named(NamedKey::F9) if pressed => {
            state.lab.show_analysis_panel = !state.lab.show_analysis_panel;
        }
        Key::Named(NamedKey::F12) if pressed => {
            state.lab.screenshot_requested = true;
            state.lab.log_event(state.world.frame, "SCREENSHOT", "Screenshot requested (F12)");
        }
        _ => {}
    }

    // Simulation controls — only if egui didn't consume the event
    if egui_consumed {
        return;
    }

    match &event.logical_key {
        Key::Named(NamedKey::Space) if pressed => {
            state.sim_params.paused = !state.sim_params.paused;
            state.lab.log_event(
                state.world.frame,
                "CONTROL",
                if state.sim_params.paused { "Paused" } else { "Resumed" },
            );
        }

        Key::Character(c) => match c.as_str() {
            "w" | "W" => state.keys.w = pressed,
            "s" | "S" => state.keys.s = pressed,
            "a" | "A" => state.keys.a = pressed,
            "d" | "D" => state.keys.d = pressed,
            "q" | "Q" => state.keys.q = pressed,
            "e" | "E" => state.keys.e = pressed,
            "r" | "R" if pressed => {
                state.lab.restart_requested = true;
            }
            "h" | "H" if pressed => {
                state.sim_params.show_extended_ui = !state.sim_params.show_extended_ui;
            }
            "1" if pressed => state.sim_params.visualization_mode = 0,
            "2" if pressed => state.sim_params.visualization_mode = 1,
            "3" if pressed => state.sim_params.visualization_mode = 2,
            "4" if pressed => state.sim_params.visualization_mode = 3,
            "5" if pressed => state.sim_params.visualization_mode = 4,
            "v" | "V" if pressed => {
                state.sim_params.vsync = !state.sim_params.vsync;
                let mode = if state.sim_params.vsync {
                    wgpu::PresentMode::AutoVsync
                } else {
                    wgpu::PresentMode::Immediate
                };
                state.surface_config.present_mode = mode;
                state.surface.configure(&state.device, &state.surface_config);
            }
            "[" if pressed => {
                state.sim_params.mutation_rate =
                    (state.sim_params.mutation_rate * 0.9).max(0.1);
            }
            "]" if pressed => {
                state.sim_params.mutation_rate =
                    (state.sim_params.mutation_rate * 1.1).min(5.0);
            }
            _ => {}
        },

        Key::Named(named) => match named {
            NamedKey::Tab if pressed => {
                state.sim_params.visualization_mode =
                    (state.sim_params.visualization_mode + 1) % VIS_MODE_COUNT;
            }
            NamedKey::ArrowUp if pressed => {
                state.sim_params.time_step =
                    (state.sim_params.time_step * 1.1).min(2.0);
            }
            NamedKey::ArrowDown if pressed => {
                state.sim_params.time_step =
                    (state.sim_params.time_step * 0.9).max(0.1);
            }
            NamedKey::ArrowRight if pressed => {
                state.sim_params.simulation_speed =
                    (state.sim_params.simulation_speed + 1).min(20);
            }
            NamedKey::ArrowLeft if pressed => {
                state.sim_params.simulation_speed =
                    state.sim_params.simulation_speed.saturating_sub(1).max(1);
            }
            _ => {}
        },

        _ => {}
    }
}

// ======================== Frame Rendering ========================

fn redraw(state: &mut AppState) {
    // FPS (exponential moving average)
    let now = Instant::now();
    let dt = now.duration_since(state.last_redraw).as_secs_f32().max(0.0001);
    state.last_redraw = now;
    state.fps = state.fps * 0.95 + (1.0 / dt) * 0.05;

    // Camera movement from held keys
    state
        .camera
        .apply_pan(state.keys.w, state.keys.s, state.keys.a, state.keys.d);
    state
        .camera
        .apply_zoom_keys(state.keys.e, state.keys.q);

    // Upload camera uniform
    state.queue.write_buffer(
        &state.pipelines.camera_buffer,
        0,
        bytemuck::bytes_of(&state.camera.uniforms()),
    );

    // Upload render params with current visualization mode
    let render_params = RenderParams {
        width: WORLD_WIDTH,
        height: WORLD_HEIGHT,
        visualization_mode: state.sim_params.visualization_mode,
        _pad: 0,
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
    state
        .egui_winit_state
        .handle_platform_output(&state.window, full_output.platform_output);

    // ---- Handle lab actions ----
    // Restart
    if state.lab.restart_requested {
        let seed = state.sim_params.effective_seed();
        state.world = WorldState::new_with_seed(&state.device, seed);
        state.pipelines =
            create_pipelines(&state.device, &state.world, state.surface_config.format);
        state.lab.restart_requested = false;
        state.last_diag = None;
        state.lab.log_event(state.world.frame, "RESTART", "Simulation restarted");
        if let Some(s) = seed {
            state.lab.log_event(state.world.frame, "SEED", &format!("Seed: {}", s));
        }
        log::info!("Simulation restarted (seed: {:?})", seed);
    }

    // ---- Handle perturbation ----
    if state.sim_params.perturbation_active {
        state.world.apply_perturbation(
            &state.device,
            &state.queue,
            &state.sim_params,
        );
        state.sim_params.perturbation_active = false;
        log::info!(
            "Perturbation applied: {} intensity={:.2} radius={:.2}",
            state.sim_params.perturbation_type.name(),
            state.sim_params.perturbation_intensity,
            state.sim_params.perturbation_radius,
        );
        state.lab.set_status(format!(
            "Perturbation '{}' applied",
            state.sim_params.perturbation_type.name(),
        ));
    }

    // Update diag interval from lab UI
    state.diag_interval = state.lab.metrics_sample_interval.max(1);

    // ---- Prepare HUD (only when Lab UI hidden, to avoid overlap) ----
    let win_w = state.surface_config.width;
    let win_h = state.surface_config.height;
    if !state.lab.show_lab_ui {
        state.hud.prepare(
            &state.device,
            &state.queue,
            &state.sim_params,
            state.world.frame,
            state.fps,
            state.camera.zoom,
            win_w,
            win_h,
        );
    }

    let dispatch_x = (WORLD_WIDTH + WORKGROUP_X - 1) / WORKGROUP_X;
    let dispatch_y = (WORLD_HEIGHT + WORKGROUP_Y - 1) / WORKGROUP_Y;
    let dispatch_linear = (total_pixels() + 255) / 256;

    // ---- Simulation steps ----
    if !state.sim_params.paused {
        let steps = state.sim_params.simulation_speed;
        for _ in 0..steps {
            state
                .world
                .update_step_uniforms_dynamic(&state.queue, &state.sim_params);

            let cur = state.world.cur();
            let mut sim_encoder = state
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("sim_encoder"),
                });
            encode_simulation_passes(
                &mut sim_encoder,
                &state.pipelines,
                cur,
                dispatch_x,
                dispatch_y,
                dispatch_linear,
            );
            state.queue.submit(std::iter::once(sim_encoder.finish()));
            state.world.swap();
        }
    } else if state.lab.step_requested {
        // Single step while paused
        state
            .world
            .update_step_uniforms_dynamic(&state.queue, &state.sim_params);
        let cur = state.world.cur();
        let mut sim_encoder = state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("step_encoder"),
            });
        encode_simulation_passes(
            &mut sim_encoder,
            &state.pipelines,
            cur,
            dispatch_x,
            dispatch_y,
            dispatch_linear,
        );
        state.queue.submit(std::iter::once(sim_encoder.finish()));
        state.world.swap();
        state.lab.step_requested = false;
        state.lab.log_event(state.world.frame, "CONTROL", "Single step");
    }

    // ---- Render pass ----
    let render_cur = 1 - state.world.cur();
    let mut encoder = state
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
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

    let view = output
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    // Simulation render pass
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.02,
                        g: 0.02,
                        b: 0.05,
                        a: 1.0,
                    }),
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

        // HUD overlay (only when Lab UI hidden)
        if !state.lab.show_lab_ui {
            state.hud.render(&mut pass);
        }
    }

    // ---- Screenshot capture (from simulation render, before egui overlay) ----
    let do_screenshot = state.lab.screenshot_requested;
    let mut screenshot_staging: Option<wgpu::Buffer> = None;
    let mut screenshot_padded_bpr: u32 = 0;

    if do_screenshot {
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let unpadded_bpr = win_w * 4;
        let padded_bpr = (unpadded_bpr + align - 1) / align * align;
        screenshot_padded_bpr = padded_bpr;

        let staging = state.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screenshot_staging"),
            size: (padded_bpr * win_h) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &output.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(win_h),
                },
            },
            wgpu::Extent3d {
                width: win_w,
                height: win_h,
                depth_or_array_layers: 1,
            },
        );
        screenshot_staging = Some(staging);
    }

    // Submit the simulation render encoder (with optional screenshot copy)
    state.queue.submit(std::iter::once(encoder.finish()));

    // ---- egui render pass (on top of simulation, separate encoder) ----
    let paint_jobs = state
        .egui_ctx
        .tessellate(full_output.shapes, full_output.pixels_per_point);

    for (id, image_delta) in &full_output.textures_delta.set {
        state
            .egui_renderer
            .update_texture(&state.device, &state.queue, *id, image_delta);
    }

    let screen_descriptor = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [win_w, win_h],
        pixels_per_point: full_output.pixels_per_point,
    };

    let mut egui_encoder = state
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("egui_encoder"),
        });

    state.egui_renderer.update_buffers(
        &state.device,
        &state.queue,
        &mut egui_encoder,
        &paint_jobs,
        &screen_descriptor,
    );

    render_egui_pass(
        &state.egui_renderer,
        &mut egui_encoder,
        &view,
        &paint_jobs,
        &screen_descriptor,
    );

    state.queue.submit(std::iter::once(egui_encoder.finish()));

    // ---- Read back screenshot ----
    if do_screenshot {
        if let Some(staging) = &screenshot_staging {
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            state.device.poll(wgpu::Maintain::Wait);

            if let Ok(Ok(())) = rx.recv() {
                let data = slice.get_mapped_range();
                // Extract RGBA data, removing row padding & swapping BGRA→RGBA
                let mut rgba = Vec::with_capacity((win_w * win_h * 4) as usize);
                for row in 0..win_h {
                    let start = (row * screenshot_padded_bpr) as usize;
                    let end = start + (win_w * 4) as usize;
                    let row_data = &data[start..end];
                    for chunk in row_data.chunks_exact(4) {
                        // BGRA → RGBA swap
                        rgba.push(chunk[2]); // R
                        rgba.push(chunk[1]); // G
                        rgba.push(chunk[0]); // B
                        rgba.push(chunk[3]); // A
                    }
                }
                drop(data);
                staging.unmap();

                match state.lab.save_screenshot(
                    state.world.frame,
                    win_w,
                    win_h,
                    &rgba,
                    state.sim_params.visualization_mode,
                ) {
                    Ok(path) => {
                        state.lab.set_status(format!("Screenshot saved: {:?}", path));
                        state.lab.log_event(
                            state.world.frame,
                            "SCREENSHOT",
                            &format!("Saved to {:?}", path),
                        );
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
            let path = state
                .lab
                .run_dir
                .join(format!("snapshot_frame{:06}.snap", state.world.frame));
            match state_io::save_snapshot(path.to_str().unwrap_or("snapshot.snap"), &snap) {
                Ok(()) => {
                    state
                        .lab
                        .set_status(format!("Snapshot saved: {:?}", path));
                    state.lab.log_event(
                        state.world.frame,
                        "SNAPSHOT",
                        &format!("Saved to {:?}", path),
                    );
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
        && state.world.frame % state.diag_interval == 0
    {
        if let Some(snap) = state.world.readback_snapshot(&state.device, &state.queue) {
            let diag = SimDiagnostics::from_snapshot(&snap);
            state
                .lab
                .record_metrics(&diag, state.world.frame, state.fps);
            diag.log(
                state.world.frame,
                target_total_mass(),
                state.last_diag.as_ref(),
            );
            state.last_diag = Some(diag);
        }
    }

    state.window.request_redraw();
}

// ======================== egui Render Helper ========================

/// Render egui paint jobs into a render pass.
/// Extracted as a free function to decouple the egui::Renderer lifetime
/// from the AppState borrow, allowing the render pass encoder to be local.
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
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    // forget_lifetime converts RenderPass<'encoder> → RenderPass<'static>
    // which is required by egui_wgpu::Renderer::render in wgpu 24.
    let mut pass = pass.forget_lifetime();
    renderer.render(&mut pass, paint_jobs, screen_descriptor);
}

// ======================== Simulation Encoding ========================

fn encode_simulation_passes(
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

    // Pass 2: Evolution (Lenia + metabolism + advection + DNA + mutations)
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("evolution_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.evolution_pipeline);
        pass.set_bind_group(0, &pipelines.evolution_bind_groups[cur], &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    // Pass 3: Resource dynamics (Gray-Scott)
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("resources_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.resources_pipeline);
        pass.set_bind_group(0, &pipelines.resources_bind_groups[cur], &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    // Pass 4a: Sum total mass (reduction)
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sum_mass_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.sum_mass_pipeline);
        pass.set_bind_group(0, &pipelines.normalize_bind_groups[cur], &[]);
        pass.dispatch_workgroups(dispatch_linear, 1, 1);
    }

    // Pass 4b: Normalize mass to target
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
