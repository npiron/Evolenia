// ============================================================================
// pipeline.rs — EvoLenia v2
// GPU pipeline creation (compute & render) and bind-group-layout helpers.
// ============================================================================

use wgpu::util::DeviceExt;

use crate::camera::CameraUniforms;
use crate::world::WorldState;

// ======================== Pipelines ========================

/// All GPU pipelines and their associated bind groups.
pub struct Pipelines {
    pub velocity_pipeline: wgpu::ComputePipeline,
    pub velocity_bind_groups: [wgpu::BindGroup; 2],

    pub evolution_pipeline: wgpu::ComputePipeline,
    pub evolution_bind_groups: [wgpu::BindGroup; 2],

    pub resources_pipeline: wgpu::ComputePipeline,
    pub resources_bind_groups: [wgpu::BindGroup; 2],

    pub sum_mass_pipeline: wgpu::ComputePipeline,
    pub normalize_pipeline: wgpu::ComputePipeline,
    pub normalize_bind_groups: [wgpu::BindGroup; 2],

    pub render_pipeline: wgpu::RenderPipeline,
    pub render_bind_groups: [wgpu::BindGroup; 2],

    pub camera_buffer: wgpu::Buffer,
}

// ======================== Pipeline Creation ========================

pub fn create_pipelines(
    device: &wgpu::Device,
    world: &WorldState,
    surface_format: wgpu::TextureFormat,
) -> Pipelines {
    // ---- Load shaders ----
    let velocity_shader = load_shader(device, "compute_velocity", include_str!("shaders/compute_velocity.wgsl"));
    let evolution_shader = load_shader(device, "compute_evolution", include_str!("shaders/compute_evolution.wgsl"));
    let resources_shader = load_shader(device, "compute_resources", include_str!("shaders/compute_resources.wgsl"));
    let normalize_shader = load_shader(device, "normalize_mass", include_str!("shaders/normalize_mass.wgsl"));
    let render_shader = load_shader(device, "render", include_str!("shaders/render.wgsl"));

    // ================================================================
    // VELOCITY PIPELINE
    // ================================================================
    let velocity_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("velocity_bgl"),
        entries: &[
            bgl_uniform(0),
            bgl_storage_ro(1),
            bgl_storage_ro(2),
            bgl_storage_rw(3),
        ],
    });

    let velocity_pipeline = create_compute_pipeline(device, "velocity", &velocity_bgl, &velocity_shader, "main");

    let velocity_bind_groups = [
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("velocity_bg_0"),
            layout: &velocity_bgl,
            entries: &[
                bg_buffer(0, &world.velocity_params_buffer),
                bg_buffer(1, &world.mass[0]),
                bg_buffer(2, &world.genome_a[0]),
                bg_buffer(3, &world.velocity),
            ],
        }),
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("velocity_bg_1"),
            layout: &velocity_bgl,
            entries: &[
                bg_buffer(0, &world.velocity_params_buffer),
                bg_buffer(1, &world.mass[1]),
                bg_buffer(2, &world.genome_a[1]),
                bg_buffer(3, &world.velocity),
            ],
        }),
    ];

    // ================================================================
    // EVOLUTION PIPELINE
    // ================================================================
    let evolution_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("evolution_bgl"),
        entries: &[
            bgl_uniform(0),
            bgl_storage_ro(1),
            bgl_storage_ro(2),
            bgl_storage_ro(3),
            bgl_storage_ro(4),
            bgl_storage_ro(5),
            bgl_storage_ro(6),
            bgl_storage_rw(7),
            bgl_storage_rw(8),
            bgl_storage_rw(9),
            bgl_storage_rw(10),
        ],
    });

    let evolution_pipeline = create_compute_pipeline(device, "evolution", &evolution_bgl, &evolution_shader, "main");

    let evolution_bind_groups = [
        // cur=0: read [0], write [1]
        device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        }),
        // cur=1: read [1], write [0]
        device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        }),
    ];

    // ================================================================
    // RESOURCES PIPELINE
    // ================================================================
    let resources_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("resources_bgl"),
        entries: &[
            bgl_uniform(0),
            bgl_storage_ro(1),
            bgl_storage_rw(2),
        ],
    });

    let resources_pipeline = create_compute_pipeline(device, "resources", &resources_bgl, &resources_shader, "main");

    // After evolution, the "next" buffer has new mass.
    // cur=0 → evolution wrote to [1], so resources reads [1]
    let resources_bind_groups = [
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resources_bg_0"),
            layout: &resources_bgl,
            entries: &[
                bg_buffer(0, &world.resource_params_buffer),
                bg_buffer(1, &world.mass[1]),
                bg_buffer(2, &world.resource_map),
            ],
        }),
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resources_bg_1"),
            layout: &resources_bgl,
            entries: &[
                bg_buffer(0, &world.resource_params_buffer),
                bg_buffer(1, &world.mass[0]),
                bg_buffer(2, &world.resource_map),
            ],
        }),
    ];

    // ================================================================
    // NORMALIZE PIPELINE (two entry points in one shader)
    // ================================================================
    let normalize_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("normalize_bgl"),
        entries: &[
            bgl_uniform(0),
            bgl_storage_rw(1),
            bgl_storage_rw(2),
        ],
    });

    let normalize_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("normalize_pipeline_layout"),
        bind_group_layouts: &[&normalize_bgl],
        push_constant_ranges: &[],
    });

    let sum_mass_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("sum_mass_pipeline"),
        layout: Some(&normalize_layout),
        module: &normalize_shader,
        entry_point: Some("sum_mass"),
        compilation_options: Default::default(),
        cache: None,
    });

    let normalize_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("normalize_pipeline"),
        layout: Some(&normalize_layout),
        module: &normalize_shader,
        entry_point: Some("normalize"),
        compilation_options: Default::default(),
        cache: None,
    });

    // cur=0 → next is [1]
    let normalize_bind_groups = [
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("normalize_bg_0"),
            layout: &normalize_bgl,
            entries: &[
                bg_buffer(0, &world.normalize_params_buffer),
                bg_buffer(1, &world.mass[1]),
                bg_buffer(2, &world.mass_sum),
            ],
        }),
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("normalize_bg_1"),
            layout: &normalize_bgl,
            entries: &[
                bg_buffer(0, &world.normalize_params_buffer),
                bg_buffer(1, &world.mass[0]),
                bg_buffer(2, &world.mass_sum),
            ],
        }),
    ];

    // ================================================================
    // RENDER PIPELINE
    // ================================================================
    let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("render_bgl"),
        entries: &[
            bgl_uniform(0),
            bgl_storage_ro(1),
            bgl_storage_ro(2),
            bgl_storage_ro(3),
            bgl_uniform(4),
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
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &render_shader,
            entry_point: Some("fs_main"),
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

    // Camera uniform buffer
    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("camera_uniforms"),
        contents: bytemuck::bytes_of(&CameraUniforms::default()),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Render reads from the "next" buffer (post-evolution, before swap)
    let render_bind_groups = [
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render_bg_0"),
            layout: &render_bgl,
            entries: &[
                bg_buffer(0, &world.render_params_buffer),
                bg_buffer(1, &world.mass[1]),
                bg_buffer(2, &world.energy[1]),
                bg_buffer(3, &world.genome_a[1]),
                bg_buffer(4, &camera_buffer),
            ],
        }),
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render_bg_1"),
            layout: &render_bgl,
            entries: &[
                bg_buffer(0, &world.render_params_buffer),
                bg_buffer(1, &world.mass[0]),
                bg_buffer(2, &world.energy[0]),
                bg_buffer(3, &world.genome_a[0]),
                bg_buffer(4, &camera_buffer),
            ],
        }),
    ];

    Pipelines {
        velocity_pipeline,
        velocity_bind_groups,
        evolution_pipeline,
        evolution_bind_groups,
        resources_pipeline,
        resources_bind_groups,
        sum_mass_pipeline,
        normalize_pipeline,
        normalize_bind_groups,
        render_pipeline,
        render_bind_groups,
        camera_buffer,
    }
}

// ======================== Helpers ========================

fn load_shader(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    name: &str,
    bgl: &wgpu::BindGroupLayout,
    module: &wgpu::ShaderModule,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{name}_pipeline_layout")),
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&format!("{name}_pipeline")),
        layout: Some(&layout),
        module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

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
        visibility: wgpu::ShaderStages::COMPUTE,
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
