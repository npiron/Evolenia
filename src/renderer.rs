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

/// All glyphon resources needed for HUD text rendering.
pub struct HudRenderer {
    pub font_system: FontSystem,
    pub swash_cache: SwashCache,
    pub glyph_viewport: GlyphViewport,
    pub text_atlas: TextAtlas,
    pub text_renderer: TextRenderer,
}

/// Configuration for HUD text preparation.
pub struct HudPrepareConfig<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub params: &'a SimulationParams,
    pub frame: u32,
    pub fps: f32,
    pub win_w: u32,
    pub win_h: u32,
    pub hud_mode: u8,
    // NES HUD metrics
    pub nes_species: usize,
    pub nes_total_mass: f32,
    pub nes_entropy: f32,
    pub nes_avg_energy: f32,
    pub nes_live_fraction: f32,
    pub nes_predator_fraction: f32,
    pub nes_prey_fraction: f32,
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
        let text_renderer = TextRenderer::new(
            &mut text_atlas,
            device,
            wgpu::MultisampleState::default(),
            None,
        );

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
    /// Returns `true` if preparation succeeded (i.e., text is ready to render).
    pub fn prepare(&mut self, config: &HudPrepareConfig<'_>) -> bool {
        self.glyph_viewport.update(
            config.queue,
            Resolution {
                width: config.win_w,
                height: config.win_h,
            },
        );

        // Mode 0: off, Mode 1: deprecated (treated as off), Mode 2: NES retro
        let (hud_text, font_size, text_color, x_pos, y_pos) = match config.hud_mode {
            2 => build_nes_hud(config),
            _ => {
                // Mode 0 or 1 — nothing to render
                return false;
            }
        };

        // Larger font for better readability (was 14.0/18.0)
        let mut text_buf = TextBuffer::new(
            &mut self.font_system,
            Metrics::new(font_size, font_size * 1.35),
        );
        text_buf.set_size(
            &mut self.font_system,
            Some(config.win_w as f32),
            Some(config.win_h as f32),
        );
        text_buf.set_text(
            &mut self.font_system,
            &hud_text,
            Attrs::new().family(Family::Monospace),
            Shaping::Basic,
        );
        text_buf.shape_until_scroll(&mut self.font_system, false);

        match self.text_renderer.prepare(
            config.device,
            config.queue,
            &mut self.font_system,
            &mut self.text_atlas,
            &self.glyph_viewport,
            [TextArea {
                buffer: &text_buf,
                left: x_pos,
                top: y_pos,
                scale: 1.0,
                bounds: TextBounds {
                    left: 0,
                    top: 0,
                    right: config.win_w as i32,
                    bottom: config.win_h as i32,
                },
                default_color: text_color,
                custom_glyphs: &[],
            }],
            &mut self.swash_cache,
        ) {
            Ok(()) => true,
            Err(e) => {
                log::warn!("HUD text preparation failed: {:?}", e);
                false
            }
        }
    }

    /// Render HUD overlay into an active render pass.
    /// Only call this after a successful `prepare()`.
    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if let Err(e) = self
            .text_renderer
            .render(&self.text_atlas, &self.glyph_viewport, pass)
        {
            log::warn!("HUD render failed: {:?}", e);
        }
    }

    /// Trim the glyph atlas after presenting.
    pub fn trim(&mut self) {
        self.text_atlas.trim();
    }
}

// ======================== NES Retro HUD ========================

/// Build a NES-style retro HUD: black background bar at the bottom with green text.
/// Returns (text, font_size, color, x_pos, y_pos).
fn build_nes_hud(config: &HudPrepareConfig<'_>) -> (String, f32, GlyphColor, f32, f32) {
    let nes_green = GlyphColor::rgb(0, 255, 60); // NES phosphor green
    let font_size = 16.0;

    let mode_name = visualization_mode_name(config.params.visualization_mode);
    let pause_str = if config.params.paused { " ⏸" } else { "" };

    let line1 = format!(
        "┌──────────────────────────────────────────────────────────────────────────────┐\n\
         │ FRAME:{:>7} │ FPS:{:>5} │ SP:{:>4} │ MASS:{:>8} │ ENT:{:>6} │ ENG:{:>6} │ LIVE:{:>4}% │\n\
         │  PRED:{:>4}% │  PREY:{:>4}% │\n\
         └──────────────────────────────────────────────────────────────────────────────┘\n\
          MODE: {}{}",
        config.frame,
        config.fps as u32,
        config.nes_species,
        config.nes_total_mass as u32,
        format!("{:.2}", config.nes_entropy),
        format!("{:.2}", config.nes_avg_energy),
        (config.nes_live_fraction * 100.0) as u32,
        (config.nes_predator_fraction * 100.0) as u32,
        (config.nes_prey_fraction * 100.0) as u32,
        mode_name,
        pause_str,
    );

    let line_count = 5.0;
    let y_pos = config.win_h as f32 - (line_count * font_size * 1.35) - 12.0;
    let x_pos = 8.0;

    (line1, font_size, nes_green, x_pos, y_pos)
}
