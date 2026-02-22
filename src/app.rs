// ============================================================================
// app.rs — EvoLenia v2
// Application state and winit event-loop handler.
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
use crate::config::SimulationParams;
use crate::input::KeysHeld;
use crate::metrics::SimDiagnostics;
use crate::pipeline::{create_pipelines, Pipelines};
use crate::renderer::HudRenderer;
use crate::world::*;

// ======================== Application ========================

pub struct App {
    state: Option<AppState>,
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

    // HUD
    hud: HudRenderer,

    // Timing
    last_redraw: Instant,
    fps: f32,

    // Diagnostics
    last_diag: Option<SimDiagnostics>,
    diag_interval: u32,
}

impl App {
    pub fn new() -> Self {
        Self { state: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window_attrs = WindowAttributes::default()
            .with_title("EvoLenia v2 — Artificial Life Engine")
            .with_inner_size(winit::dpi::LogicalSize::new(1024u32, 1024u32));

        let window = Arc::new(event_loop.create_window(window_attrs).unwrap());

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let (device, queue, surface_config) =
            pollster::block_on(init_gpu(&instance, &surface, &window));

        surface.configure(&device, &surface_config);

        let world = WorldState::new(&device);
        let pipelines = create_pipelines(&device, &world, surface_config.format);
        let hud = HudRenderer::new(&device, &queue, surface_config.format);

        log::info!(
            "EvoLenia v2 initialized: {}x{}, target mass = {:.0}",
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
            last_redraw: Instant::now(),
            fps: 0.0,
            last_diag: None,
            diag_interval: 300,
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

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput { event, .. } => {
                handle_keyboard(state, event_loop, &event);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match &delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                };
                state.camera.apply_scroll(scroll);
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
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
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
) {
    let pressed = event.state.is_pressed();

    match &event.logical_key {
        Key::Named(NamedKey::Escape) if pressed => event_loop.exit(),

        Key::Named(NamedKey::Space) if pressed => {
            state.sim_params.paused = !state.sim_params.paused;
            log::info!(
                "Simulation {}",
                if state.sim_params.paused {
                    "PAUSED"
                } else {
                    "RESUMED"
                }
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
                log::info!("Restarting simulation...");
                state.world = WorldState::new(&state.device);
                state.pipelines =
                    create_pipelines(&state.device, &state.world, state.surface_config.format);
            }
            "h" | "H" if pressed => {
                state.sim_params.show_extended_ui = !state.sim_params.show_extended_ui;
                log::info!(
                    "Extended HUD: {}",
                    if state.sim_params.show_extended_ui {
                        "ON"
                    } else {
                        "OFF"
                    }
                );
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
                log::info!(
                    "VSync: {}",
                    if state.sim_params.vsync { "ON" } else { "OFF" }
                );
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
            NamedKey::ArrowUp if pressed => {
                state.sim_params.time_step = (state.sim_params.time_step * 1.1).min(2.0);
            }
            NamedKey::ArrowDown if pressed => {
                state.sim_params.time_step = (state.sim_params.time_step * 0.9).max(0.1);
            }
            NamedKey::ArrowRight if pressed => {
                state.sim_params.simulation_speed =
                    (state.sim_params.simulation_speed + 1).min(10);
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

    // Prepare HUD
    let win_w = state.surface_config.width;
    let win_h = state.surface_config.height;
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

    let dispatch_x = (WORLD_WIDTH + WORKGROUP_X - 1) / WORKGROUP_X;
    let dispatch_y = (WORLD_HEIGHT + WORKGROUP_Y - 1) / WORKGROUP_Y;
    let dispatch_linear = (total_pixels() + 255) / 256;

    // Run N simulation steps per frame (multi-step for higher throughput)
    if !state.sim_params.paused {
        let steps = state.sim_params.simulation_speed;
        for _ in 0..steps {
            state.world.update_uniforms(&state.queue);
            state.queue.write_buffer(&state.world.mass_sum, 0, &[0u8; 8]);

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
    }

    // ---- Render pass (once per frame, reads latest simulation data) ----
    // After multi-step, world.cur() is up to date. render_bind_groups[1-cur]
    // reads the buffer that was last written to.
    let render_cur = 1 - state.world.cur();
    let mut encoder = state
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_encoder"),
        });

    // Render pass
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

        // HUD overlay
        state.hud.render(&mut pass);
    }

    state.queue.submit(std::iter::once(encoder.finish()));
    output.present();
    state.hud.trim();

    // Periodic diagnostics via GPU readback
    if !state.sim_params.paused && state.world.frame > 0 && state.world.frame % state.diag_interval == 0 {
        if let Some(snap) = state.world.readback_snapshot(&state.device, &state.queue) {
            let diag = SimDiagnostics::from_snapshot(&snap);
            diag.log(state.world.frame, target_total_mass(), state.last_diag.as_ref());
            state.last_diag = Some(diag);
        } else {
            log::warn!("Frame {} | GPU readback failed", state.world.frame);
        }
    }

    state.window.request_redraw();
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
