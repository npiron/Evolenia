// ============================================================================
// app/mod.rs — EvoLenia v2
// Application state and winit event-loop handler with egui UI integration.
// ============================================================================

mod gpu;
pub mod input;
mod render;
pub mod theme;

use std::sync::Arc;
use std::time::Instant;

use winit::{
    application::ApplicationHandler,
    event::{MouseScrollDelta, WindowEvent},
    window::{Window, WindowAttributes},
};

use crate::camera::CameraState;
use crate::config::SimulationParams;
use crate::lab::LabState;
use crate::pipeline::{create_pipelines, Pipelines};
use crate::renderer::HudRenderer;
use crate::state_io;
use crate::world::*;

pub use theme::apply_theme;

// ======================== Application ========================

pub struct App {
    state: Option<AppState>,
    config: AppConfig,
}

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub initial_state_path: Option<String>,
    pub diag_interval: u32,
    pub sim_params: SimulationParams,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            initial_state_path: None,
            diag_interval: 300,
            sim_params: SimulationParams::default(),
        }
    }
}

pub(crate) fn recreate_msaa(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("msaa_texture"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 4,
        dimension: wgpu::TextureDimension::D2,
        format: config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

pub(crate) struct AppState {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub msaa_texture: Option<wgpu::Texture>,
    pub msaa_view: Option<wgpu::TextureView>,

    pub world: WorldState,
    pub pipelines: Pipelines,

    pub window: Arc<Window>,

    pub camera: CameraState,
    pub keys: input::KeysHeld,
    pub sim_params: SimulationParams,

    pub hud: HudRenderer,

    pub egui_ctx: egui::Context,
    pub egui_winit_state: egui_winit::State,
    pub egui_renderer: egui_wgpu::Renderer,

    pub lab: LabState,

    pub last_redraw: Instant,
    pub fps: f32,

    pub last_diag: Option<crate::metrics::SimDiagnostics>,
    pub diag_interval: u32,
}

impl App {
    pub fn new(config: AppConfig) -> Self {
        Self {
            state: None,
            config,
        }
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
            pollster::block_on(gpu::init_gpu(&instance, &surface, &window));

        surface.configure(&device, &surface_config);

        let (msaa_tex, msaa_view) = recreate_msaa(&device, &surface_config);

        let sim_params = self.config.sim_params.clone();
        let mut world =
            WorldState::new_with_config(&device, sim_params.effective_seed(), &sim_params);
        if let Some(path) = &self.config.initial_state_path {
            match state_io::load_snapshot(path) {
                Ok(snapshot) => {
                    if world.apply_snapshot(&queue, &snapshot) {
                        log::info!("Loaded simulation state from {}", path);
                    } else {
                        log::warn!(
                            "State file {} has incompatible dimensions; using fresh world",
                            path
                        );
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
        theme::apply_default_theme(&egui_ctx);

        let egui_winit_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            event_loop,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        let egui_renderer =
            egui_wgpu::Renderer::new(&device, surface_config.format, None, 1, false);

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
            msaa_texture: Some(msaa_tex),
            msaa_view: Some(msaa_view),
            world,
            pipelines,
            window: window.clone(),
            camera: CameraState::default(),
            keys: input::KeysHeld::default(),
            sim_params,
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

        let egui_response = state
            .egui_winit_state
            .on_window_event(&state.window, &event);

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput { event, .. } => {
                input::handle_keyboard(state, event_loop, &event, egui_response.consumed);
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
                    state
                        .surface
                        .configure(&state.device, &state.surface_config);
                    let (tex, view) = recreate_msaa(&state.device, &state.surface_config);
                    state.msaa_texture = Some(tex);
                    state.msaa_view = Some(view);
                }
            }

            WindowEvent::RedrawRequested => {
                render::redraw(state);
            }

            _ => {}
        }
    }
}
