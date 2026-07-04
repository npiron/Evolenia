// ============================================================================
// lab_ui/logs.rs — Bottom panel showing events log
// ============================================================================

use crate::lab::LabState;

pub fn render_bottom_logs_panel(ctx: &egui::Context, lab: &mut LabState) {
    egui::TopBottomPanel::bottom("logs_panel")
        .default_height(120.0)
        .min_height(60.0)
        .max_height(300.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("📋 Events Log").strong());
                ui.label(format!("({} events)", lab.events.len()));
                if ui.button("Clear").clicked() {
                    lab.events.clear();
                }
                if ui.button("Export").clicked() {
                    match lab.export_events_log() {
                        Ok(path) => lab.set_status(format!("Exported events to {:?}", path)),
                        Err(e) => lab.set_status(format!("Export failed: {}", e)),
                    }
                }
            });
            ui.separator();
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for event in lab.events.iter().rev().take(100) {
                        let color = match event.event_type.as_str() {
                            "PARAM_CHANGE" => egui::Color32::from_rgb(200, 130, 20),
                            "RUN_START" | "RUN_END" => egui::Color32::from_rgb(30, 160, 40),
                            "CONTROL" => egui::Color32::from_rgb(20, 100, 200),
                            "SCREENSHOT" | "SNAPSHOT" => egui::Color32::from_rgb(140, 40, 200),
                            _ => egui::Color32::from_rgb(100, 100, 110),
                        };
                        ui.label(
                            egui::RichText::new(event.to_log_line())
                                .small()
                                .color(color)
                                .monospace(),
                        );
                    }
                });
        });
}
