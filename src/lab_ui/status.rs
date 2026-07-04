// ============================================================================
// lab_ui/status.rs — Floating status bar at the bottom
// ============================================================================

use crate::lab::LabState;

pub fn render_status_bar(ctx: &egui::Context, lab: &mut LabState) {
    if let Some(msg) = lab.current_status() {
        let msg = msg.to_string();
        egui::Area::new(egui::Id::new("status_bar"))
            .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(0.0, -10.0))
            .show(ctx, |ui| {
                egui::Frame::default()
                    .fill(egui::Color32::from_rgba_premultiplied(220, 245, 220, 240))
                    .corner_radius(egui::CornerRadius::same(4))
                    .inner_margin(egui::Margin::symmetric(12, 6))
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new(msg).color(egui::Color32::from_rgb(20, 80, 20)),
                        );
                    });
            });
    }
}
