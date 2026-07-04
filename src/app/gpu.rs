// ============================================================================
// app/gpu.rs — GPU adapter and device initialization
// ============================================================================

use winit::window::Window;

pub async fn init_gpu(
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

    let present_mode = if surface_caps
        .present_modes
        .contains(&wgpu::PresentMode::Mailbox)
    {
        log::info!("Present mode: Mailbox (uncapped FPS)");
        wgpu::PresentMode::Mailbox
    } else if surface_caps
        .present_modes
        .contains(&wgpu::PresentMode::Immediate)
    {
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
