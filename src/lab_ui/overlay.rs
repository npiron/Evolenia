// ============================================================================
// lab_ui/overlay.rs — Minimal overlay shown when Lab UI is hidden
// ============================================================================

use crate::config::SimulationParams;
use crate::lab::LabState;

use super::dashboard::predator_context;

pub fn render_minimal_overlay(ctx: &egui::Context, params: &SimulationParams, lab: &mut LabState) {
    let last = lab.metrics_history.last();
    let species = last.map_or(0, |m| m.species);
    let live_frac = last.map_or(0.0, |m| m.live_fraction * 100.0);
    let predator_frac = last.map_or(0.0, |m| m.predator_fraction * 100.0);
    let prey_frac = last.map_or(0.0, |m| m.prey_fraction * 100.0);
    let avg_energy = last.map_or(0.0, |m| m.avg_energy);
    let total_mass = last.map_or(0.0, |m| m.total_mass);
    let mass_drift = last.map_or(0.0, |m| m.mass_drift_pct);
    let entropy = last.map_or(0.0, |m| m.entropy);
    let frame = last.map_or(0, |m| m.frame);
    let fps = last.map_or(0.0, |m| m.fps);

    let (pred_status, pred_color) = predator_context(lab);

    egui::Area::new(egui::Id::new("minimal_overlay"))
        .fixed_pos(egui::pos2(16.0, 16.0))
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(egui::Color32::from_rgba_premultiplied(8, 8, 16, 248))
                .corner_radius(10)
                .inner_margin(egui::Margin::symmetric(18, 14))
                .stroke(egui::Stroke::new(
                    1.5,
                    egui::Color32::from_rgba_premultiplied(255, 255, 255, 90),
                ))
                .show(ui, |ui| {
                    // ── Row 1: Title + Pause indicator ──
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("🌌 EvoLenia")
                                .size(18.0)
                                .strong()
                                .color(egui::Color32::from_rgb(140, 200, 255)),
                        );
                        if params.paused {
                            ui.add_space(10.0);
                            ui.label(
                                egui::RichText::new("⏸ PAUSED")
                                    .size(14.0)
                                    .strong()
                                    .color(egui::Color32::from_rgb(255, 160, 60)),
                            );
                        }
                    });

                    ui.add_space(8.0);

                    // ── Row 2: Key metrics ──
                    ui.horizontal(|ui| {
                        hud_metric(ui, "🧬 Espèces", &format!("{}", species), egui::Color32::from_rgb(200, 160, 255));
                        ui.add_space(16.0);
                        hud_metric(ui, "💚 Vie", &format!("{:.0}%", live_frac), egui::Color32::from_rgb(80, 255, 120));
                        ui.add_space(16.0);
                        hud_metric(ui, "🦁 Prédateurs", &format!("{:.0}%", predator_frac), egui::Color32::from_rgb(255, 160, 70));
                        ui.add_space(16.0);
                        hud_metric(ui, "🐑 Proies", &format!("{:.0}%", prey_frac), egui::Color32::from_rgb(120, 220, 255));
                    });

                    ui.add_space(6.0);

                    // ── Row 3: Secondary metrics ──
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(format!("⚡ Énergie: {:.3}", avg_energy))
                                .size(12.0)
                                .color(egui::Color32::from_rgb(230, 230, 120)),
                        );
                        ui.add_space(16.0);
                        ui.label(
                            egui::RichText::new(format!("📦 Masse: {:.0}", total_mass))
                                .size(12.0)
                                .color(egui::Color32::from_rgb(210, 210, 235)),
                        );
                        if mass_drift.abs() > 1.0 {
                            let drift_color = if mass_drift > 0.0 {
                                egui::Color32::from_rgb(120, 220, 120)
                            } else {
                                egui::Color32::from_rgb(220, 140, 120)
                            };
                            ui.label(
                                egui::RichText::new(format!("  ({:+.1}%)", mass_drift))
                                    .size(10.0)
                                    .color(drift_color),
                            );
                        }
                        ui.add_space(16.0);
                        ui.label(
                            egui::RichText::new(format!("🌀 Entropie: {:.2}", entropy))
                                .size(12.0)
                                .color(egui::Color32::from_rgb(200, 220, 245)),
                        );
                    });

                    // ── Predator trend ──
                    if !pred_status.is_empty() {
                        ui.add_space(4.0);
                        ui.label(
                            egui::RichText::new(pred_status)
                                .size(11.0)
                                .color(pred_color),
                        );
                    }

                    ui.add_space(6.0);
                    ui.separator();
                    ui.add_space(4.0);

                    // ── Row 4: Technical info + shortcuts ──
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(format!("Frame {}  •  FPS {:.0}", frame, fps))
                                .size(10.0)
                                .monospace()
                                .color(egui::Color32::from_rgb(130, 130, 150)),
                        );
                        ui.add_space(16.0);
                        ui.label(
                            egui::RichText::new("F1:Lab  Space:Pause  H:HUD  WASD:Pan  Q/E:Zoom")
                                .size(10.0)
                                .color(egui::Color32::from_rgb(120, 120, 145)),
                        );
                    });
                });
        });
}

fn hud_metric(ui: &mut egui::Ui, label: &str, value: &str, color: egui::Color32) {
    ui.vertical(|ui| {
        ui.label(
            egui::RichText::new(label)
                .size(10.0)
                .color(egui::Color32::from_rgb(170, 170, 190)),
        );
        ui.label(
            egui::RichText::new(value)
                .size(16.0)
                .strong()
                .monospace()
                .color(color),
        );
    });
}
