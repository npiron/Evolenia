// ============================================================================
// renderer.rs — EvoLenia v2
// HUD text rendering via glyphon and GPU render pass orchestration.
// ============================================================================

use glyphon::{
    Attrs, Buffer as TextBuffer, Cache as GlyphCache, Color as GlyphColor, Family, FontSystem,
    Metrics, Resolution, Shaping, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer,
    Viewport as GlyphViewport,
};

use crate::config::{visualization_mode_name, SimulationParams};
use crate::world::{target_total_mass, WORLD_HEIGHT, WORLD_WIDTH};

/// All glyphon resources needed for HUD text rendering.
pub struct HudRenderer {
    pub font_system: FontSystem,
    pub swash_cache: SwashCache,
    pub glyph_viewport: GlyphViewport,
    pub text_atlas: TextAtlas,
    pub text_renderer: TextRenderer,
}

impl HudRenderer {
    /// Initialize the HUD text rendering subsystem.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let mut font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let glyph_cache = GlyphCache::new(device);
        let glyph_viewport = GlyphViewport::new(device, &glyph_cache);
        let mut text_atlas = TextAtlas::new(device, queue, &glyph_cache, surface_format);
        let text_renderer =
            TextRenderer::new(&mut text_atlas, device, wgpu::MultisampleState::default(), None);

        // Prime font system so first frame renders correctly
        let mut primer = TextBuffer::new(&mut font_system, Metrics::new(16.0, 20.0));
        primer.set_text(
            &mut font_system,
            "EvoLenia",
            Attrs::new().family(Family::Monospace),
            Shaping::Basic,
        );

        Self {
            font_system,
            swash_cache,
            glyph_viewport,
            text_atlas,
            text_renderer,
        }
    }

    /// Prepare HUD text for the current frame.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        params: &SimulationParams,
        frame: u32,
        fps: f32,
        camera_zoom: f32,
        win_w: u32,
        win_h: u32,
    ) {
        self.glyph_viewport.update(
            queue,
            Resolution {
                width: win_w,
                height: win_h,
            },
        );

        let hud_text = build_hud_text(params, frame, fps, camera_zoom);

        let mut text_buf = TextBuffer::new(&mut self.font_system, Metrics::new(14.0, 18.0));
        text_buf.set_size(&mut self.font_system, Some(win_w as f32), Some(win_h as f32));
        text_buf.set_text(
            &mut self.font_system,
            &hud_text,
            Attrs::new().family(Family::Monospace),
            Shaping::Basic,
        );
        text_buf.shape_until_scroll(&mut self.font_system, false);

        self.text_renderer
            .prepare(
                device,
                queue,
                &mut self.font_system,
                &mut self.text_atlas,
                &self.glyph_viewport,
                [TextArea {
                    buffer: &text_buf,
                    left: 10.0,
                    top: 10.0,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: 0,
                        top: 0,
                        right: win_w as i32,
                        bottom: win_h as i32,
                    },
                    default_color: GlyphColor::rgb(220, 220, 220),
                    custom_glyphs: &[],
                }],
                &mut self.swash_cache,
            )
            .unwrap();
    }

    /// Render HUD overlay into an active render pass.
    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        self.text_renderer
            .render(&self.text_atlas, &self.glyph_viewport, pass)
            .unwrap();
    }

    /// Trim the glyph atlas after presenting.
    pub fn trim(&mut self) {
        self.text_atlas.trim();
    }
}

// ======================== HUD Text Builder ========================

fn build_hud_text(params: &SimulationParams, frame: u32, fps: f32, camera_zoom: f32) -> String {
    let pause_status = if params.paused { " [PAUSED]" } else { "" };

    if params.show_extended_ui {
        format!(
            "━━━ EvoLenia v2.0 — Extended HUD ━━━\n\
             Frame: {}   FPS: {:.0}{}  |  Zoom: {:.2}x\n\
             \n\
             VISUALIZATION (1-5 / Tab):\n\
             • Current: {} (<)✓(>)\n\
             • 1: Species Color  2: Energy  3: Mass  4: Diversity  5: Predator/Prey\n\
             \n\
             SIMULATION CONTROL:\n\
             • Space: {}  |  R: Restart  |  H: Toggle HUD  |  ESC: Quit\n\
             • Speed: {}x (←/→ to adjust)  |  TimeStep: {:.2}x (↑/↓)\n\
             • Mutation Rate: {:.2}x ([/] to adjust)\n\
             \n\
             CAMERA:\n\
             • Pan: WASD  |  Zoom: Q/E or Mouse Wheel\n\
             • VSync: {} (V to toggle)\n\
             \n\
             WORLD: {}×{}  |  Target Mass: {:.0}",
            frame,
            fps,
            pause_status,
            camera_zoom,
            visualization_mode_name(params.visualization_mode),
            if params.paused { "Resume" } else { "Pause" },
            params.simulation_speed,
            params.time_step,
            params.mutation_rate,
            if params.vsync { "ON" } else { "OFF" },
            WORLD_WIDTH,
            WORLD_HEIGHT,
            target_total_mass()
        )
    } else {
        format!(
            "Frame: {}   FPS: {:.0}{}   Zoom: {:.2}x\n\
             Mode: {} (1-5/Tab) | Space: Pause | R: Restart | H: Help",
            frame,
            fps,
            pause_status,
            camera_zoom,
            visualization_mode_name(params.visualization_mode),
        )
    }
}
