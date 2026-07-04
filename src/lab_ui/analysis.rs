// ============================================================================
// lab_ui/analysis.rs — Right analysis panel with stats, plots, and run comparison
// ============================================================================

use egui_plot::{Line, Plot, PlotPoints};

use crate::lab::{LabState, MetricsRecord};

pub fn render_right_analysis_panel(ctx: &egui::Context, lab: &mut LabState) {
    egui::SidePanel::right("analysis_panel")
        .default_width(380.0)
        .min_width(300.0)
        .max_width(550.0)
        .show(ctx, |ui| {
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("📈").size(24.0));
                ui.add_space(8.0);
                ui.label(
                    egui::RichText::new("Live Analysis")
                        .size(20.0)
                        .strong()
                        .color(egui::Color32::from_rgb(30, 140, 50)),
                );
            });
            ui.add_space(8.0);
            ui.separator();

            if lab.metrics_history.is_empty() {
                ui.label("No metrics data yet. Wait for diagnostics readback.");
                return;
            }

            // Live stats table
            if let Some(last) = lab.metrics_history.last() {
                let available_width = ui.available_width();
                egui::Grid::new("live_stats")
                    .num_columns(2)
                    .min_col_width(available_width / 2.0 - 8.0)
                    .striped(true)
                    .show(ui, |ui| {
                        stat_row(ui, "Frame", &format!("{}", last.frame));
                        stat_row(ui, "FPS", &format!("{:.0}", last.fps));
                        stat_row(ui, "Total Mass", &format!("{:.0}", last.total_mass));
                        stat_row(ui, "Avg Energy", &format!("{:.4}", last.avg_energy));
                        stat_row(ui, "Entropy", &format!("{:.2} bits", last.entropy));
                        stat_row(ui, "Species", &format!("{}", last.species));
                        stat_row(ui, "Live Pixels", &format!("{} ({:.1}%)", last.live_pixels, last.live_fraction * 100.0));
                        stat_row(ui, "Predators", &format!("{:.1}%", last.predator_fraction * 100.0));
                        stat_row(ui, "Avg Resource", &format!("{:.3}", last.avg_resource));
                        stat_row(ui, "Mass StdDev", &format!("{:.4}", last.mass_std_dev));
                        stat_row(ui, "Prey %", &format!("{:.1}%", last.prey_fraction * 100.0));
                        stat_row(ui, "Opportunist %", &format!("{:.1}%", last.opportunist_fraction * 100.0));
                        stat_row(ui, "Eff. Diversity", &format!("{:.2}", last.effective_diversity));
                        stat_row(ui, "Genome Var", &format!("{:.4}", last.genome_variance));
                        stat_row(ui, "Total Energy", &format!("{:.0}", last.total_energy));
                        stat_row(ui, "Energy Flux", &format!("{:.4}", last.energy_flux));
                    });
            }
            ui.separator();

            // Time-series plots
            egui::ScrollArea::vertical().show(ui, |ui| {
                render_plot(ui, "Total Mass", &lab.metrics_history, |m| m.total_mass as f64);
                render_plot(ui, "Avg Energy", &lab.metrics_history, |m| m.avg_energy as f64);
                render_plot(ui, "Genetic Entropy", &lab.metrics_history, |m| m.entropy as f64);
                render_plot(ui, "Species Count", &lab.metrics_history, |m| m.species as f64);
                render_plot(ui, "Live Pixels", &lab.metrics_history, |m| m.live_pixels as f64);
                render_plot(ui, "FPS", &lab.metrics_history, |m| m.fps as f64);

                render_plot(ui, "Effective Diversity", &lab.metrics_history, |m| m.effective_diversity as f64);
                render_plot(ui, "Energy Flux", &lab.metrics_history, |m| m.energy_flux as f64);
                render_plot(ui, "Genome Variance", &lab.metrics_history, |m| m.genome_variance as f64);

                // Comparison section
                if !lab.completed_runs.is_empty() {
                    ui.separator();
                    ui.heading("🔀 Run Comparison");
                    render_comparison_ui(ui, lab);
                }
            });
        });
}

fn stat_row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.label(egui::RichText::new(label).color(egui::Color32::from_rgb(100, 100, 115)));
    ui.label(
        egui::RichText::new(value)
            .monospace()
            .strong()
            .color(egui::Color32::from_rgb(30, 30, 40)),
    );
    ui.end_row();
}

fn render_plot<F>(ui: &mut egui::Ui, title: &str, history: &[MetricsRecord], value_fn: F)
where
    F: Fn(&MetricsRecord) -> f64,
{
    let points: PlotPoints = history
        .iter()
        .map(|m| [m.frame as f64, value_fn(m)])
        .collect();

    let plot = Plot::new(format!("plot_{}", title))
        .height(160.0)
        .show_axes(true)
        .show_grid(true)
        .allow_drag(true)
        .allow_scroll(true)
        .allow_zoom(true)
        .auto_bounds([true; 2]);

    let hovered = plot.show(ui, |plot_ui| {
        plot_ui.line(Line::new(points).name(title));
    });

    if let Some(pointer) = hovered.response.hover_pos() {
        let coord = hovered.transform.value_from_position(pointer);
        ui.label(
            egui::RichText::new(format!(
                "{} — frame {}: {:.4}",
                title, coord.x as u32, coord.y,
            ))
            .small()
            .monospace()
            .color(egui::Color32::from_rgb(80, 80, 100)),
        );
    }

    ui.label(egui::RichText::new(title).small().strong());
    ui.add_space(4.0);
}

// ======================== Comparison UI ========================

fn render_comparison_ui(ui: &mut egui::Ui, lab: &mut LabState) {
    ui.horizontal(|ui| {
        ui.label("Run A:");
        egui::ComboBox::from_id_salt("comp_a")
            .selected_text(
                lab.comparison_a
                    .and_then(|i| lab.completed_runs.get(i))
                    .map_or("Select…".to_string(), |r| r.run_id.clone()),
            )
            .show_ui(ui, |ui| {
                for (i, run) in lab.completed_runs.iter().enumerate() {
                    ui.selectable_value(&mut lab.comparison_a, Some(i), &run.run_id);
                }
            });
    });

    ui.horizontal(|ui| {
        ui.label("Run B:");
        egui::ComboBox::from_id_salt("comp_b")
            .selected_text(
                lab.comparison_b
                    .and_then(|i| lab.completed_runs.get(i))
                    .map_or("Select…".to_string(), |r| r.run_id.clone()),
            )
            .show_ui(ui, |ui| {
                for (i, run) in lab.completed_runs.iter().enumerate() {
                    ui.selectable_value(&mut lab.comparison_b, Some(i), &run.run_id);
                }
            });
    });

    if let (Some(a_idx), Some(b_idx)) = (lab.comparison_a, lab.comparison_b) {
        if a_idx != b_idx {
            if let (Some(run_a), Some(run_b)) =
                (lab.completed_runs.get(a_idx), lab.completed_runs.get(b_idx))
            {
                let csv_a = run_a.run_dir.join("metrics.csv");
                let csv_b = run_b.run_dir.join("metrics.csv");

                match (
                    LabState::load_comparison_metrics(&csv_a),
                    LabState::load_comparison_metrics(&csv_b),
                ) {
                    (Ok(metrics_a), Ok(metrics_b)) => {
                        render_comparison_plot(ui, "Mass", &metrics_a, &metrics_b, |m| {
                            m.total_mass as f64
                        });
                        render_comparison_plot(ui, "Entropy", &metrics_a, &metrics_b, |m| {
                            m.entropy as f64
                        });
                        render_comparison_plot(ui, "Species", &metrics_a, &metrics_b, |m| {
                            m.species as f64
                        });
                    }
                    _ => {
                        ui.label("Could not load comparison data.");
                    }
                }
            }
        }
    }
}

fn render_comparison_plot<F>(
    ui: &mut egui::Ui,
    title: &str,
    a: &[MetricsRecord],
    b: &[MetricsRecord],
    value_fn: F,
) where
    F: Fn(&MetricsRecord) -> f64,
{
    let points_a: PlotPoints = a.iter().map(|m| [m.frame as f64, value_fn(m)]).collect();
    let points_b: PlotPoints = b.iter().map(|m| [m.frame as f64, value_fn(m)]).collect();

    Plot::new(format!("comp_{}", title))
        .height(140.0)
        .show_axes(true)
        .allow_drag(true)
        .allow_scroll(true)
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new(points_a)
                    .name("Run A")
                    .color(egui::Color32::from_rgb(100, 200, 255)),
            );
            plot_ui.line(
                Line::new(points_b)
                    .name("Run B")
                    .color(egui::Color32::from_rgb(255, 150, 100)),
            );
        });
    ui.label(
        egui::RichText::new(format!("{} (A vs B)", title))
            .small()
            .strong(),
    );
    ui.add_space(4.0);
}
