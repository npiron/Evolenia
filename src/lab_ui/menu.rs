// ============================================================================
// lab_ui/menu.rs — Menu bar (Fichier, Affichage, Simulation, Aide)
// ============================================================================

use crate::config::{visualization_mode_name, SimulationParams, VIS_MODE_COUNT};
use crate::lab::LabState;

pub fn render_menu_bar(ctx: &egui::Context, params: &mut SimulationParams, lab: &mut LabState) {
    egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
        egui::menu::bar(ui, |ui| {
            // ── Fichier ──
            ui.menu_button("Fichier", |ui| {
                if ui.button("📊 Export Metrics CSV").clicked() {
                    match lab.export_metrics_csv() {
                        Ok(path) => lab.set_status(format!("Exporté: {:?}", path)),
                        Err(e) => lab.set_status(format!("Échec: {}", e)),
                    }
                    ui.close_menu();
                }
                if ui.button("📝 Export Rapport").clicked() {
                    match lab.export_report(params) {
                        Ok(path) => lab.set_status(format!("Rapport: {:?}", path)),
                        Err(e) => lab.set_status(format!("Échec: {}", e)),
                    }
                    ui.close_menu();
                }
                ui.separator();
                if ui.button("📷 Screenshot (F12)").clicked() {
                    lab.screenshot_requested = true;
                    ui.close_menu();
                }
                if ui.button("💾 Snapshot").clicked() {
                    lab.snapshot_requested = true;
                    ui.close_menu();
                }
                ui.separator();
                if ui.button("❌ Quitter").clicked() {
                    std::process::exit(0);
                }
            });

            // ── Affichage ──
            ui.menu_button("Affichage", |ui| {
                if ui
                    .checkbox(&mut lab.show_analysis_panel, "Panneau d'analyse (F9)")
                    .clicked()
                {
                    ui.close_menu();
                }
                if ui
                    .checkbox(&mut lab.show_logs_panel, "Journal d'événements")
                    .clicked()
                {
                    ui.close_menu();
                }
                if ui.checkbox(&mut lab.dark_mode, "🌙 Dark Mode").clicked() {
                    crate::app::apply_theme(ctx, lab.dark_mode);
                    ui.close_menu();
                }
                ui.separator();
                if ui
                    .checkbox(&mut lab.show_legend, "🎨 Légende (canvas)")
                    .clicked()
                {
                    ui.close_menu();
                }
                ui.separator();
                ui.label("Visualisation:");
                for mode in 0..VIS_MODE_COUNT {
                    let name = visualization_mode_name(mode);
                    if ui
                        .radio_value(&mut params.visualization_mode, mode, name)
                        .clicked()
                    {
                        ui.close_menu();
                    }
                }
                ui.separator();
                if ui.checkbox(&mut params.vsync, "VSync").clicked() {
                    ui.close_menu();
                }
            });

            // ── Simulation ──
            ui.menu_button("Simulation", |ui| {
                let btn_text = if params.paused {
                    "▶ Play"
                } else {
                    "⏸ Pause"
                };
                if ui.button(btn_text).clicked() {
                    params.paused = !params.paused;
                    lab.log_event(
                        0,
                        "CONTROL",
                        if params.paused { "Paused" } else { "Resumed" },
                    );
                    ui.close_menu();
                }
                if ui.button("⏭ Step").clicked() {
                    lab.step_requested = true;
                    params.paused = true;
                    ui.close_menu();
                }
                if ui.button("🔄 Restart").clicked() {
                    lab.restart_requested = true;
                    ui.close_menu();
                }
                ui.separator();
                ui.label(format!("Vitesse: ×{}", params.simulation_speed));
                ui.add(
                    egui::Slider::new(&mut params.simulation_speed, 1..=20)
                        .text("")
                        .step_by(1.0),
                );
            });

            // ── Aide ──
            ui.menu_button("Aide", |ui| {
                ui.label(egui::RichText::new("Raccourcis clavier").strong());
                ui.label("F1    — Afficher/Cacher le Lab");
                ui.label("F9    — Panneau d'analyse");
                ui.label("F12   — Screenshot");
                ui.label("Space — Pause/Play");
                ui.label("R     — Restart");
                ui.label("H     — HUD (cycle)");
                ui.label("WASD  — Pan caméra");
                ui.label("Q/E   — Zoom");
                ui.label("1-5   — Mode de visualisation");
                ui.label("↑/↓   — Time step");
                ui.label("←/→   — Vitesse simulation");
            });
        });
    });
}
