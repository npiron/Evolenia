// ============================================================================
// app/theme.rs — egui theme management
// ============================================================================

/// Apply either light or dark theme to the egui context.
pub fn apply_theme(ctx: &egui::Context, dark_mode: bool) {
    ctx.set_visuals(if dark_mode {
        egui::Visuals::dark()
    } else {
        egui::Visuals::light()
    });
}

/// Apply the default macOS-inspired light theme with custom fonts and spacing.
pub fn apply_default_theme(egui_ctx: &egui::Context) {
    // macOS-inspired light theme
    let mut visuals = egui::Visuals::light();
    visuals.window_fill = egui::Color32::from_rgb(236, 236, 240);
    visuals.panel_fill = egui::Color32::from_rgb(246, 246, 248);
    visuals.extreme_bg_color = egui::Color32::from_rgb(220, 220, 226);
    visuals.faint_bg_color = egui::Color32::from_rgb(250, 250, 252);
    visuals.window_stroke =
        egui::Stroke::new(1.0, egui::Color32::from_rgba_premultiplied(0, 0, 0, 30));
    visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(232, 232, 237);
    visuals.widgets.noninteractive.fg_stroke =
        egui::Stroke::new(1.0, egui::Color32::from_rgb(60, 60, 70));
    visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(220, 220, 226);
    visuals.widgets.inactive.fg_stroke =
        egui::Stroke::new(1.0, egui::Color32::from_rgb(40, 40, 50));
    visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(200, 220, 245);
    visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.5, egui::Color32::from_rgb(0, 80, 180));
    visuals.widgets.active.bg_fill = egui::Color32::from_rgb(0, 122, 255);
    visuals.widgets.active.fg_stroke = egui::Stroke::new(2.0, egui::Color32::WHITE);
    visuals.selection.bg_fill = egui::Color32::from_rgba_premultiplied(0, 122, 255, 60);
    visuals.selection.stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(0, 122, 255));
    visuals.hyperlink_color = egui::Color32::from_rgb(0, 100, 200);
    egui_ctx.set_visuals(visuals);

    // Larger default font size for better readability
    let mut style = (*egui_ctx.style()).clone();
    style.text_styles.insert(
        egui::TextStyle::Body,
        egui::FontId::new(15.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Button,
        egui::FontId::new(15.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Heading,
        egui::FontId::new(20.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Monospace,
        egui::FontId::new(14.0, egui::FontFamily::Monospace),
    );
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.button_padding = egui::vec2(10.0, 6.0);
    egui_ctx.set_style(style);
}
