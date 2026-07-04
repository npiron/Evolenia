// ============================================================================
// lab_ui/dashboard.rs — Live dashboard with key stats, gauges, and controls
// ============================================================================

use crate::lab::LabState;

pub fn render_dashboard(
    ui: &mut egui::Ui,
    params: &mut crate::config::SimulationParams,
    lab: &mut LabState,
) {
    let frame = lab.metrics_history.last().map_or(0, |m| m.frame);
    let fps = lab.metrics_history.last().map_or(0.0, |m| m.fps);
    let species = lab.metrics_history.last().map_or(0, |m| m.species);
    let live_frac = lab
        .metrics_history
        .last()
        .map_or(0.0, |m| m.live_fraction * 100.0);
    let predator_frac = lab
        .metrics_history
        .last()
        .map_or(0.0, |m| m.predator_fraction * 100.0);

    egui::Frame::new()
        .fill(egui::Color32::from_rgb(242, 244, 248))
        .corner_radius(8)
        .inner_margin(egui::Margin::symmetric(12, 10))
        .stroke(egui::Stroke::new(
            1.0,
            egui::Color32::from_rgb(200, 210, 220),
        ))
        .show(ui, |ui| {
            // Row 1: Key stats
            ui.horizontal(|ui| {
                dashboard_stat(
                    ui,
                    "🕐 Frame",
                    &format!("{}", frame),
                    egui::Color32::from_rgb(20, 150, 80),
                );
                ui.separator();
                dashboard_stat(
                    ui,
                    "⚡ FPS",
                    &format!("{:.0}", fps),
                    egui::Color32::from_rgb(200, 120, 20),
                );
                ui.separator();
                dashboard_stat(
                    ui,
                    "🧬 Sp.",
                    &format!("{}", species),
                    egui::Color32::from_rgb(130, 40, 190),
                );
            });
            ui.add_space(6.0);

            // Row 2: Gauges
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Vie")
                        .size(11.0)
                        .color(egui::Color32::from_rgb(100, 100, 115)),
                );
                ui.add(
                    egui::ProgressBar::new(live_frac / 100.0)
                        .desired_width(80.0)
                        .text(format!("{:.0}%", live_frac))
                        .fill(egui::Color32::from_rgb(40, 180, 90)),
                );
                ui.add_space(8.0);
                ui.label(
                    egui::RichText::new("Prédateurs")
                        .size(11.0)
                        .color(egui::Color32::from_rgb(100, 100, 115)),
                );
                ui.add(
                    egui::ProgressBar::new(predator_frac / 100.0)
                        .desired_width(80.0)
                        .text(format!("{:.0}%", predator_frac))
                        .fill(egui::Color32::from_rgb(220, 100, 60)),
                );
                let (pred_status, pred_color) = predator_context(lab);
                ui.label(egui::RichText::new(pred_status).size(9.0).color(pred_color));
            });

            ui.add_space(8.0);
            ui.separator();
            ui.add_space(4.0);

            // Row 3: Play/Pause + Step + Speed + Restart
            ui.horizontal(|ui| {
                let (btn_text, btn_color) = if params.paused {
                    ("▶ Play", egui::Color32::from_rgb(40, 160, 60))
                } else {
                    ("⏸ Pause", egui::Color32::from_rgb(220, 140, 40))
                };
                let play_btn = egui::Button::new(egui::RichText::new(btn_text).strong().size(14.0))
                    .fill(btn_color)
                    .min_size(egui::vec2(75.0, 28.0));
                if ui
                    .add(play_btn)
                    .on_hover_text("Pause/Play (Space)")
                    .clicked()
                {
                    params.paused = !params.paused;
                    lab.log_event(
                        0,
                        "CONTROL",
                        if params.paused { "Paused" } else { "Resumed" },
                    );
                }
                if ui
                    .add(egui::Button::new("⏭").min_size(egui::vec2(28.0, 28.0)))
                    .on_hover_text("Step one frame (while paused)")
                    .clicked()
                {
                    lab.step_requested = true;
                    params.paused = true;
                }
                if ui
                    .add(egui::Button::new("🔄").min_size(egui::vec2(28.0, 28.0)))
                    .on_hover_text("Restart simulation with current params")
                    .clicked()
                {
                    lab.restart_requested = true;
                }
            });

            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Vitesse")
                        .size(12.0)
                        .color(egui::Color32::from_rgb(90, 90, 105)),
                );
                let speed_slider = egui::Slider::new(&mut params.simulation_speed, 1..=20)
                    .text("×")
                    .step_by(1.0);
                ui.add(speed_slider);
            });
        });
}

fn dashboard_stat(ui: &mut egui::Ui, label: &str, value: &str, color: egui::Color32) {
    ui.vertical(|ui| {
        ui.label(
            egui::RichText::new(label)
                .size(10.0)
                .color(egui::Color32::from_rgb(100, 100, 115)),
        );
        ui.label(egui::RichText::new(value).size(16.0).strong().color(color));
    });
}

/// Determine predator population context from recent metrics history.
pub fn predator_context(lab: &LabState) -> (&'static str, egui::Color32) {
    let history = &lab.metrics_history;
    if history.len() < 3 {
        return ("collecting data…", egui::Color32::from_rgb(140, 140, 155));
    }

    let recent: Vec<f32> = history
        .iter()
        .rev()
        .take(20)
        .map(|m| m.predator_fraction)
        .collect();
    let current = recent[0];
    let avg_20 = recent.iter().sum::<f32>() / recent.len() as f32;

    if current < 0.001 && avg_20 < 0.001 {
        ("no predators yet", egui::Color32::from_rgb(140, 140, 155))
    } else if current < 0.01 && avg_20 > 0.02 {
        (
            "⚠ population collapsed",
            egui::Color32::from_rgb(220, 140, 30),
        )
    } else if current < 0.01 {
        ("near extinction", egui::Color32::from_rgb(200, 150, 60))
    } else if recent.len() >= 8 {
        let trend: f32 = recent.windows(4).map(|w| w[0] - w[3]).sum();
        if trend > 0.03 {
            ("↑ rising", egui::Color32::from_rgb(40, 180, 90))
        } else if trend < -0.03 {
            ("↓ declining", egui::Color32::from_rgb(220, 100, 60))
        } else {
            ("• stable", egui::Color32::from_rgb(80, 160, 80))
        }
    } else {
        ("monitoring…", egui::Color32::from_rgb(140, 140, 155))
    }
}
