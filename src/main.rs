// ============================================================================
// main.rs — EvoLenia v2
// WGPU initialization, compute/render pipeline creation, and event loop.
// ============================================================================

mod world;

use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::EventLoop,
    keyboard::{Key, NamedKey},
    window::{Window, WindowAttributes},
};
use world::*;

// ======================== Pipelines ========================

struct Pipelines {
    velocity_pipeline: wgpu::ComputePipeline,
    velocity_bind_group: wgpu::BindGroup,
    velocity_bind_group_alt: wgpu::BindGroup,

    evolution_pipeline: wgpu::ComputePipeline,
    evolution_bind_groups: [wgpu::BindGroup; 2], // [cur=0, cur=1]

    resources_pipeline: wgpu::ComputePipeline,
    resources_bind_groups: [wgpu::BindGroup; 2],

    sum_mass_pipeline: wgpu::ComputePipeline,
    normalize_pipeline: wgpu::ComputePipeline,
    normalize_bind_groups: [wgpu::BindGroup; 2],

    render_pipeline: wgpu::RenderPipeline,
    render_bind_groups: [wgpu::BindGroup; 2],
}

fn create_pipelines(device: &wgpu::Device, world: &WorldState, surface_format: wgpu::TextureFormat) -> Pipelines {
    // ---- Load shaders ----
    let velocity_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("compute_velocity"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/compute_velocity.wgsl").into()),
    });
    let evolution_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("compute_evolution"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/compute_evolution.wgsl").into()),
    });
    let resources_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("compute_resources"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/compute_resources.wgsl").into()),
    });
    let normalize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("normalize_mass"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/normalize_mass.wgsl").into()),
    });
    let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("render"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/render.wgsl").into()),
    });

    // ================================================================
    // VELOCITY PIPELINE
    // ================================================================
    let velocity_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("velocity_bgl"),
        entries: &[
            bgl_uniform(0),            // params
            bgl_storage_ro(1),          // mass_current
            bgl_storage_ro(2),          // genome_a_current
            bgl_storage_rw(3),          // velocity
        ],
    });

    let velocity_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("velocity_pipeline_layout"),
        bind_group_layouts: &[&velocity_bgl],
        push_constant_ranges: &[],
    });

    let velocity_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("velocity_pipeline"),
        layout: Some(&velocity_pipeline_layout),
        module: &velocity_shader,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });

    // Two bind groups for ping-pong
    let velocity_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("velocity_bg_0"),
        layout: &velocity_bgl,
        entries: &[
            bg_buffer(0, &world.velocity_params_buffer),
            bg_buffer(1, &world.mass[0]),
            bg_buffer(2, &world.genome_a[0]),
            bg_buffer(3, &world.velocity),
        ],
    });

    let velocity_bind_group_alt = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("velocity_bg_1"),
        layout: &velocity_bgl,
        entries: &[
            bg_buffer(0, &world.velocity_params_buffer),
            bg_buffer(1, &world.mass[1]),
            bg_buffer(2, &world.genome_a[1]),
            bg_buffer(3, &world.velocity),
        ],
    });

    // ================================================================
    // EVOLUTION PIPELINE
    // ================================================================
    let evolution_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("evolution_bgl"),
        entries: &[
            bgl_uniform(0),     // params
            bgl_storage_ro(1),  // mass_in
            bgl_storage_ro(2),  // energy_in
            bgl_storage_ro(3),  // genome_a_in
            bgl_storage_ro(4),  // genome_b_in
            bgl_storage_ro(5),  // resource_map
            bgl_storage_ro(6),  // velocity
            bgl_storage_rw(7),  // mass_out
            bgl_storage_rw(8),  // energy_out
            bgl_storage_rw(9),  // genome_a_out
            bgl_storage_rw(10), // genome_b_out
        ],
    });

    let evolution_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("evolution_pipeline_layout"),
        bind_group_layouts: &[&evolution_bgl],
        push_constant_ranges: &[],
    });

    let evolution_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("evolution_pipeline"),
        layout: Some(&evolution_pipeline_layout),
        module: &evolution_shader,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });

    // cur=0: read from [0], write to [1]
    let evolution_bg_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("evolution_bg_0"),
        layout: &evolution_bgl,
        entries: &[
            bg_buffer(0, &world.sim_params_buffer),
            bg_buffer(1, &world.mass[0]),
            bg_buffer(2, &world.energy[0]),
            bg_buffer(3, &world.genome_a[0]),
            bg_buffer(4, &world.genome_b[0]),
            bg_buffer(5, &world.resource_map),
            bg_buffer(6, &world.velocity),
            bg_buffer(7, &world.mass[1]),
            bg_buffer(8, &world.energy[1]),
            bg_buffer(9, &world.genome_a[1]),
            bg_buffer(10, &world.genome_b[1]),
        ],
    });

    // cur=1: read from [1], write to [0]
    let evolution_bg_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("evolution_bg_1"),
        layout: &evolution_bgl,
        entries: &[
            bg_buffer(0, &world.sim_params_buffer),
            bg_buffer(1, &world.mass[1]),
            bg_buffer(2, &world.energy[1]),
            bg_buffer(3, &world.genome_a[1]),
            bg_buffer(4, &world.genome_b[1]),
            bg_buffer(5, &world.resource_map),
            bg_buffer(6, &world.velocity),
            bg_buffer(7, &world.mass[0]),
            bg_buffer(8, &world.energy[0]),
            bg_buffer(9, &world.genome_a[0]),
            bg_buffer(10, &world.genome_b[0]),
        ],
    });

    let evolution_bind_groups = [evolution_bg_0, evolution_bg_1];

    // ================================================================
    // RESOURCES PIPELINE
    // ================================================================
    let resources_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("resources_bgl"),
        entries: &[
            bgl_uniform(0),    // params
            bgl_storage_ro(1), // mass (read the NEXT mass after evolution)
            bgl_storage_rw(2), // resource_map
        ],
    });

    let resources_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("resources_pipeline_layout"),
        bind_group_layouts: &[&resources_bgl],
        push_constant_ranges: &[],
    });

    let resources_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("resources_pipeline"),
        layout: Some(&resources_pipeline_layout),
        module: &resources_shader,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });

    // After evolution, the "next" buffer has new mass.
    // cur=0 → evolution wrote to [1], so resources reads [1]
    let resources_bg_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("resources_bg_0"),
        layout: &resources_bgl,
        entries: &[
            bg_buffer(0, &world.resource_params_buffer),
            bg_buffer(1, &world.mass[1]),
            bg_buffer(2, &world.resource_map),
        ],
    });

    let resources_bg_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("resources_bg_1"),
        layout: &resources_bgl,
        entries: &[
            bg_buffer(0, &world.resource_params_buffer),
            bg_buffer(1, &world.mass[0]),
            bg_buffer(2, &world.resource_map),
        ],
    });

    let resources_bind_groups = [resources_bg_0, resources_bg_1];

    // ================================================================
    // NORMALIZE PIPELINE (two entry points in one shader)
    // ================================================================
    let normalize_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("normalize_bgl"),
        entries: &[
            bgl_uniform(0),    // params
            bgl_storage_rw(1), // mass (read-write for normalization)
            bgl_storage_rw(2), // mass_sum (atomic)
        ],
    });

    let normalize_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("normalize_pipeline_layout"),
        bind_group_layouts: &[&normalize_bgl],
        push_constant_ranges: &[],
    });

    let sum_mass_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("sum_mass_pipeline"),
        layout: Some(&normalize_pipeline_layout),
        module: &normalize_shader,
        entry_point: "sum_mass",
        compilation_options: Default::default(),
        cache: None,
    });

    let normalize_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("normalize_pipeline"),
        layout: Some(&normalize_pipeline_layout),
        module: &normalize_shader,
        entry_point: "normalize",
        compilation_options: Default::default(),
        cache: None,
    });

    // Normalize operates on the "next" buffer (post-evolution)
    // cur=0 → next is [1]
    let normalize_bg_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("normalize_bg_0"),
        layout: &normalize_bgl,
        entries: &[
            bg_buffer(0, &world.normalize_params_buffer),
            bg_buffer(1, &world.mass[1]),
            bg_buffer(2, &world.mass_sum),
        ],
    });

    let normalize_bg_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("normalize_bg_1"),
        layout: &normalize_bgl,
        entries: &[
            bg_buffer(0, &world.normalize_params_buffer),
            bg_buffer(1, &world.mass[0]),
            bg_buffer(2, &world.mass_sum),
        ],
    });

    let normalize_bind_groups = [normalize_bg_0, normalize_bg_1];

    // ================================================================
    // RENDER PIPELINE
    // ================================================================
    let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("render_bgl"),
        entries: &[
            bgl_uniform(0),    // render_params
            bgl_storage_ro(1), // mass
            bgl_storage_ro(2), // genome_a
        ],
    });

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("render_pipeline_layout"),
        bind_group_layouts: &[&render_bgl],
        push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("render_pipeline"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &render_shader,
            entry_point: "vs_main",
            buffers: &[], // No vertex buffer — full-screen quad from vertex_index
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &render_shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    // Render reads whichever buffer was just written (the "next" which becomes "current" after swap)
    // But we render BEFORE swap, so we read from "next" = 1-current
    // cur=0 → render from [1]  (post-evolution data)
    let render_bg_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("render_bg_0"),
        layout: &render_bgl,
        entries: &[
            bg_buffer(0, &world.render_params_buffer),
            bg_buffer(1, &world.mass[1]),
            bg_buffer(2, &world.genome_a[1]),
        ],
    });

    let render_bg_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("render_bg_1"),
        layout: &render_bgl,
        entries: &[
            bg_buffer(0, &world.render_params_buffer),
            bg_buffer(1, &world.mass[0]),
            bg_buffer(2, &world.genome_a[0]),
        ],
    });

    let render_bind_groups = [render_bg_0, render_bg_1];

    Pipelines {
        velocity_pipeline,
        velocity_bind_group,
        velocity_bind_group_alt,
        evolution_pipeline,
        evolution_bind_groups,
        resources_pipeline,
        resources_bind_groups,
        sum_mass_pipeline,
        normalize_pipeline,
        normalize_bind_groups,
        render_pipeline,
        render_bind_groups,
    }
}

// ======================== Helper Functions ========================

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bg_buffer(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

// ======================== Application ========================

struct App {
    state: Option<AppState>,
}

struct AppState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    world: WorldState,
    pipelines: Pipelines,
    window: Arc<Window>,
}

impl App {
    fn new() -> Self {
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

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let (adapter, device, queue) = pollster::block_on(async {
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: Some(&surface),
                    force_fallback_adapter: false,
                })
                .await
                .expect("Failed to find a suitable GPU adapter");

            log::info!("GPU: {}", adapter.get_info().name);

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("evolenia_device"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits {
                            max_storage_buffers_per_shader_stage: 12,
                            max_storage_buffer_binding_size: 1024 * 1024 * 1024,
                            ..Default::default()
                        },
                        memory_hints: Default::default(),
                    },
                    None,
                )
                .await
                .expect("Failed to create device");

            (adapter, device, queue)
        });

        let size = window.inner_size();
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let world = WorldState::new(&device);
        let pipelines = create_pipelines(&device, &world, surface_format);

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
            window,
        });
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
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state.is_pressed() {
                    if let Key::Named(NamedKey::Escape) = event.logical_key {
                        event_loop.exit();
                    }
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
                // Update uniform buffers with current frame counter
                state.world.update_uniforms(&state.queue);

                let cur = state.world.cur();
                let _nxt = state.world.next();

                // ---- Encode compute + render passes ----
                let mut encoder = state.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("frame_encoder"),
                    },
                );

                let dispatch_x = (WORLD_WIDTH + WORKGROUP_X - 1) / WORKGROUP_X;
                let dispatch_y = (WORLD_HEIGHT + WORKGROUP_Y - 1) / WORKGROUP_Y;
                let dispatch_linear =
                    (total_pixels() + 255) / 256;

                // Pass 1: Compute velocity field
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("velocity_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&state.pipelines.velocity_pipeline);
                    let bg = if cur == 0 {
                        &state.pipelines.velocity_bind_group
                    } else {
                        &state.pipelines.velocity_bind_group_alt
                    };
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
                }

                // Pass 2: Evolution (Lenia + metabolism + advection + DNA + mutations)
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("evolution_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&state.pipelines.evolution_pipeline);
                    pass.set_bind_group(0, &state.pipelines.evolution_bind_groups[cur], &[]);
                    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
                }

                // Pass 3: Resource dynamics (Gray-Scott)
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("resources_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&state.pipelines.resources_pipeline);
                    pass.set_bind_group(0, &state.pipelines.resources_bind_groups[cur], &[]);
                    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
                }

                // Pass 4a: Sum total mass (reduction)
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("sum_mass_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&state.pipelines.sum_mass_pipeline);
                    pass.set_bind_group(0, &state.pipelines.normalize_bind_groups[cur], &[]);
                    pass.dispatch_workgroups(dispatch_linear, 1, 1);
                }

                // Pass 4b: Normalize mass to target
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("normalize_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&state.pipelines.normalize_pipeline);
                    pass.set_bind_group(0, &state.pipelines.normalize_bind_groups[cur], &[]);
                    pass.dispatch_workgroups(dispatch_linear, 1, 1);
                }

                // Pass 5: Render
                {
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
                        // Render from "next" buffer (post-evolution, before swap)
                        pass.set_bind_group(0, &state.pipelines.render_bind_groups[cur], &[]);
                        pass.draw(0..6, 0..1); // 6 vertices = 2 triangles
                    }

                    state.queue.submit(std::iter::once(encoder.finish()));
                    output.present();
                }

                // Swap ping-pong buffers
                state.world.swap();

                // Metrics logging (every 300 frames)
                if state.world.frame % 300 == 0 {
                    log::info!("Frame {}", state.world.frame);
                }

                // Request next frame
                state.window.request_redraw();
            }
            _ => {}
        }
    }
}

// ======================== Entry Point ========================

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
