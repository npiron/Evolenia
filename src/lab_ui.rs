// ============================================================================
// lab_ui.rs — EvoLenia v2 Research Lab UI
// Complete egui-based interface for simulation control, parameter tuning,
// metrics visualization, experiment management, and data export.
// ============================================================================

use egui_plot::{Line, Plot, PlotPoints};

use crate::config::{visualization_mode_name, SimulationParams, VIS_MODE_COUNT};
use crate::lab::LabState;
use crate::world::{target_total_mass, WORLD_HEIGHT, WORLD_WIDTH};

/// Main entry point for rendering all Research Lab UI panels.
pub fn render_lab_ui(
    ctx: &egui::Context,
    params: &mut SimulationParams,
    lab: &mut LabState,
) {
    if !lab.show_lab_ui {
        // Minimal overlay when UI is hidden
        render_minimal_overlay(ctx, params, lab);
        return;
    }

    render_left_panel(ctx, params, lab);

    if lab.show_analysis_panel {
        render_right_analysis_panel(ctx, lab);
    }

    if lab.show_logs_panel {
        render_bottom_logs_panel(ctx, lab);
    }

    // Status bar
    render_status_bar(ctx, lab);
}

// ======================== Minimal Overlay ========================

fn render_minimal_overlay(
    ctx: &egui::Context,
    params: &SimulationParams,
    lab: &mut LabState,
) {
    egui::Area::new(egui::Id::new("minimal_overlay"))
        .fixed_pos(egui::pos2(10.0, 10.0))
        .show(ctx, |ui| {
            ui.visuals_mut().override_text_color = Some(egui::Color32::from_rgb(220, 220, 220));
            let pause_str = if params.paused { " [PAUSED]" } else { "" };
            ui.label(
                egui::RichText::new(format!(
                    "F: {}  FPS: {:.0}{}  | F1: Lab UI",
                    lab.metrics_history.last().map_or(0, |m| m.frame),
                    lab.metrics_history.last().map_or(0.0, |m| m.fps),
                    pause_str,
                ))
                .monospace()
                .size(13.0),
            );
        });
}

// ======================== Left Panel ========================

fn render_left_panel(
    ctx: &egui::Context,
    params: &mut SimulationParams,
    lab: &mut LabState,
) {
    egui::SidePanel::left("lab_panel")
        .default_width(280.0)
        .min_width(240.0)
        .max_width(400.0)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("🔬 EvoLenia Research Lab");
                ui.separator();

                render_control_section(ui, params, lab);
                ui.separator();
                render_params_section(ui, params, lab);
                ui.separator();
                render_visualization_section(ui, params);
                ui.separator();
                render_experiment_section(ui, params, lab);
                ui.separator();
                render_capture_section(ui, params, lab);
                ui.separator();
                render_view_toggles(ui, lab);

                ui.add_space(10.0);
            });
        });
}

// ======================== Control Section ========================

fn render_control_section(
    ui: &mut egui::Ui,
    params: &mut SimulationParams,
    lab: &mut LabState,
) {
    ui.collapsing("▶ Control", |ui| {
        ui.horizontal(|ui| {
            let play_label = if params.paused { "▶ Play" } else { "⏸ Pause" };
            if ui.button(play_label).clicked() {
                params.paused = !params.paused;
                lab.log_event(0, "CONTROL", if params.paused { "Paused" } else { "Resumed" });
            }
            if ui.button("⏭ Step").clicked() {
                lab.step_requested = true;
                params.paused = true;
            }
            if ui.button("🔄 Restart").clicked() {
                lab.restart_requested = true;
            }
        });

        ui.add_space(4.0);

        ui.horizontal(|ui| {
            ui.label("Speed:");
            if ui.add(egui::Slider::new(&mut params.simulation_speed, 1..=20).suffix("x")).changed() {
                lab.log_event(0, "PARAM_CHANGE", &format!("speed={}", params.simulation_speed));
            }
        });

        ui.horizontal(|ui| {
            ui.label("Time Step:");
            if ui.add(egui::Slider::new(&mut params.time_step, 0.1..=2.0).step_by(0.05)).changed() {
                lab.log_event(0, "PARAM_CHANGE", &format!("time_step={:.2}", params.time_step));
            }
        });

        ui.horizontal(|ui| {
            ui.label("Diag interval:");
            ui.add(egui::DragValue::new(&mut lab.metrics_sample_interval).range(10..=5000));
        });

        // Effective values
        ui.add_space(2.0);
        ui.label(
            egui::RichText::new(format!(
                "Effective dt: {:.4}  |  Steps/frame: {}",
                0.1 * params.time_step,
                params.simulation_speed,
            ))
            .small()
            .color(egui::Color32::from_rgb(150, 200, 150)),
        );
    });
}

// ======================== Parameters Section ========================

fn render_params_section(
    ui: &mut egui::Ui,
    params: &mut SimulationParams,
    lab: &mut LabState,
) {
    ui.collapsing("🧬 Simulation Parameters", |ui| {
        ui.group(|ui| {
            ui.label(egui::RichText::new("Evolution / Mutation").strong());
            if ui.add(
                egui::Slider::new(&mut params.mutation_rate, 0.1..=5.0)
                    .text("Mutation Rate")
                    .step_by(0.1),
            ).changed() {
                lab.log_event(0, "PARAM_CHANGE", &format!("mutation_rate={:.1}", params.mutation_rate));
            }
        });

        ui.group(|ui| {
            ui.label(egui::RichText::new("Predation").strong());
            if ui.add(
                egui::Slider::new(&mut params.predation_factor, 0.0..=3.0)
                    .text("Predation Factor")
                    .step_by(0.1),
            ).changed() {
                lab.log_event(0, "PARAM_CHANGE", &format!("predation={:.1}", params.predation_factor));
            }
        });

        ui.group(|ui| {
            ui.label(egui::RichText::new("Resources (Gray-Scott)").strong());
            if ui.add(
                egui::Slider::new(&mut params.resource_diffusion, 0.0..=0.5)
                    .text("Diffusion")
                    .step_by(0.01),
            ).changed() {
                lab.log_event(0, "PARAM_CHANGE", &format!("diffusion={:.3}", params.resource_diffusion));
            }
            if ui.add(
                egui::Slider::new(&mut params.resource_feed_rate, 0.0..=0.1)
                    .text("Feed Rate")
                    .step_by(0.001),
            ).changed() {
                lab.log_event(0, "PARAM_CHANGE", &format!("feed_rate={:.4}", params.resource_feed_rate));
            }
            if ui.add(
                egui::Slider::new(&mut params.resource_consumption, 0.0..=0.3)
                    .text("Consumption")
                    .step_by(0.01),
            ).changed() {
                lab.log_event(0, "PARAM_CHANGE", &format!("consumption={:.3}", params.resource_consumption));
            }
        });

        ui.group(|ui| {
            ui.label(egui::RichText::new("Mass Normalization").strong());
            if ui.checkbox(&mut params.mass_normalization_enabled, "Enabled").changed() {
                lab.log_event(0, "PARAM_CHANGE", &format!("norm_enabled={}", params.mass_normalization_enabled));
            }
            if params.mass_normalization_enabled {
                if ui.add(
                    egui::Slider::new(&mut params.mass_damping, 0.05..=1.0)
                        .text("Damping")
                        .step_by(0.05),
                ).changed() {
                    lab.log_event(0, "PARAM_CHANGE", &format!("damping={:.2}", params.mass_damping));
                }
                if ui.add(
                    egui::Slider::new(&mut params.target_mass_multiplier, 0.1..=3.0)
                        .text("Target Mass ×")
                        .step_by(0.1),
                ).changed() {
                    lab.log_event(0, "PARAM_CHANGE", &format!("target_mass_mult={:.1}", params.target_mass_multiplier));
                }
                ui.label(
                    egui::RichText::new(format!(
                        "Target: {:.0}",
                        target_total_mass() * params.target_mass_multiplier
                    ))
                    .small()
                    .color(egui::Color32::from_rgb(150, 200, 150)),
                );
            }
        });

        ui.group(|ui| {
            ui.label(egui::RichText::new("Initial Conditions (on restart)").strong());
            ui.add(
                egui::Slider::new(&mut params.num_seed_clusters, 5..=100)
                    .text("Seed Clusters"),
            );
            ui.add(
                egui::Slider::new(&mut params.seed_cluster_size, 0.5..=3.0)
                    .text("Cluster Scale")
                    .step_by(0.1),
            );
            ui.add(
                egui::Slider::new(&mut params.initial_mass_fill, 0.05..=0.5)
                    .text("Mass Fill %")
                    .step_by(0.01),
            );
        });
    });
}

// ======================== Visualization Section ========================

fn render_visualization_section(ui: &mut egui::Ui, params: &mut SimulationParams) {
    ui.collapsing("🎨 Visualization", |ui| {
        for mode in 0..VIS_MODE_COUNT {
            let name = visualization_mode_name(mode);
            if ui.radio_value(&mut params.visualization_mode, mode, name).clicked() {
                log::info!("Visualization mode: {}", name);
            }
        }
        ui.add_space(4.0);
        ui.checkbox(&mut params.vsync, "VSync");

        ui.label(
            egui::RichText::new(format!("World: {}×{}", WORLD_WIDTH, WORLD_HEIGHT))
                .small()
                .color(egui::Color32::GRAY),
        );
    });
}

// ======================== Experiment Section ========================

fn render_experiment_section(
    ui: &mut egui::Ui,
    params: &mut SimulationParams,
    lab: &mut LabState,
) {
    ui.collapsing("🧪 Experiments", |ui| {
        // Seed control
        ui.group(|ui| {
            ui.label(egui::RichText::new("Reproducibility").strong());
            ui.checkbox(&mut params.use_fixed_seed, "Use fixed seed");
            if params.use_fixed_seed {
                ui.horizontal(|ui| {
                    ui.label("Seed:");
                    ui.add(egui::DragValue::new(&mut params.fixed_seed_value).range(0..=u64::MAX));
                });
            }
            if let Some(seed) = params.effective_seed() {
                ui.label(
                    egui::RichText::new(format!("Active seed: {}", seed))
                        .small()
                        .color(egui::Color32::from_rgb(150, 200, 150)),
                );
            }
        });

        // Run management
        ui.group(|ui| {
            ui.label(egui::RichText::new("Run Management").strong());
            ui.label(format!("Run ID: {}", lab.run_id));

            ui.horizontal(|ui| {
                if ui.button("📁 Start Run").clicked() {
                    lab.start_run(params);
                }
                if ui.button("⏹ Finalize Run").clicked() {
                    lab.finalize_run(params);
                }
            });

            if lab.run_active {
                ui.label(
                    egui::RichText::new("● Recording")
                        .color(egui::Color32::from_rgb(100, 255, 100)),
                );
            }

            ui.label(format!("Metrics: {} samples", lab.metrics_history.len()));
        });

        // Presets
        ui.group(|ui| {
            ui.label(egui::RichText::new("Presets").strong());

            ui.horizontal(|ui| {
                ui.text_edit_singleline(&mut lab.preset_name);
                if ui.button("Save").clicked() {
                    save_preset(&lab.preset_name, params);
                    lab.set_status(format!("Preset '{}' saved", lab.preset_name));
                }
            });
            if ui.button("Load preset…").clicked() {
                if let Some(loaded) = load_preset(&lab.preset_name) {
                    *params = loaded;
                    lab.set_status(format!("Preset '{}' loaded", lab.preset_name));
                }
            }
            if ui.button("Reset to defaults").clicked() {
                let vis = params.visualization_mode;
                *params = SimulationParams::default();
                params.visualization_mode = vis;
                lab.set_status("Parameters reset to defaults".to_string());
            }
        });
    });
}

// ======================== Capture Section ========================

fn render_capture_section(
    ui: &mut egui::Ui,
    params: &SimulationParams,
    lab: &mut LabState,
) {
    ui.collapsing("📸 Capture", |ui| {
        ui.horizontal(|ui| {
            if ui.button("📷 Screenshot (F12)").clicked() {
                lab.screenshot_requested = true;
            }
            if ui.button("💾 Snapshot").clicked() {
                lab.snapshot_requested = true;
            }
        });

        if ui.button("📊 Export Metrics CSV").clicked() {
            match lab.export_metrics_csv() {
                Ok(path) => lab.set_status(format!("Exported to {:?}", path)),
                Err(e) => lab.set_status(format!("Export failed: {}", e)),
            }
        }

        if ui.button("📝 Export Report").clicked() {
            match lab.export_report(params) {
                Ok(path) => lab.set_status(format!("Report saved to {:?}", path)),
                Err(e) => lab.set_status(format!("Report failed: {}", e)),
            }
        }
    });
}

// ======================== View Toggles ========================

fn render_view_toggles(ui: &mut egui::Ui, lab: &mut LabState) {
    ui.collapsing("📊 View", |ui| {
        ui.checkbox(&mut lab.show_analysis_panel, "Analysis panel (F9)");
        ui.checkbox(&mut lab.show_logs_panel, "Logs panel");
    });
}

// ======================== Right Analysis Panel ========================

fn render_right_analysis_panel(ctx: &egui::Context, lab: &mut LabState) {
    egui::SidePanel::right("analysis_panel")
        .default_width(340.0)
        .min_width(250.0)
        .max_width(500.0)
        .show(ctx, |ui| {
            ui.heading("📈 Analysis");
            ui.separator();

            if lab.metrics_history.is_empty() {
                ui.label("No metrics data yet. Wait for diagnostics readback.");
                return;
            }

            // Live stats table
            if let Some(last) = lab.metrics_history.last() {
                egui::Grid::new("live_stats")
                    .num_columns(2)
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
    ui.label(label);
    ui.label(egui::RichText::new(value).monospace());
    ui.end_row();
}

fn render_plot<F>(
    ui: &mut egui::Ui,
    title: &str,
    history: &[crate::lab::MetricsRecord],
    value_fn: F,
) where
    F: Fn(&crate::lab::MetricsRecord) -> f64,
{
    let points: PlotPoints = history
        .iter()
        .map(|m| [m.frame as f64, value_fn(m)])
        .collect();

    Plot::new(format!("plot_{}", title))
        .height(100.0)
        .show_axes(true)
        .show_grid(true)
        .allow_drag(false)
        .allow_scroll(false)
        .show(ui, |plot_ui| {
            plot_ui.line(Line::new(points).name(title));
        });
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
            if let (Some(run_a), Some(run_b)) = (
                lab.completed_runs.get(a_idx),
                lab.completed_runs.get(b_idx),
            ) {
                let csv_a = run_a.run_dir.join("metrics.csv");
                let csv_b = run_b.run_dir.join("metrics.csv");

                match (
                    LabState::load_comparison_metrics(&csv_a),
                    LabState::load_comparison_metrics(&csv_b),
                ) {
                    (Ok(metrics_a), Ok(metrics_b)) => {
                        render_comparison_plot(ui, "Mass", &metrics_a, &metrics_b, |m| m.total_mass as f64);
                        render_comparison_plot(ui, "Entropy", &metrics_a, &metrics_b, |m| m.entropy as f64);
                        render_comparison_plot(ui, "Species", &metrics_a, &metrics_b, |m| m.species as f64);
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
    a: &[crate::lab::MetricsRecord],
    b: &[crate::lab::MetricsRecord],
    value_fn: F,
) where
    F: Fn(&crate::lab::MetricsRecord) -> f64,
{
    let points_a: PlotPoints = a.iter().map(|m| [m.frame as f64, value_fn(m)]).collect();
    let points_b: PlotPoints = b.iter().map(|m| [m.frame as f64, value_fn(m)]).collect();

    Plot::new(format!("comp_{}", title))
        .height(100.0)
        .show_axes(true)
        .allow_drag(false)
        .allow_scroll(false)
        .show(ui, |plot_ui| {
            plot_ui.line(Line::new(points_a).name("Run A").color(egui::Color32::from_rgb(100, 200, 255)));
            plot_ui.line(Line::new(points_b).name("Run B").color(egui::Color32::from_rgb(255, 150, 100)));
        });
    ui.label(egui::RichText::new(format!("{} (A vs B)", title)).small().strong());
    ui.add_space(4.0);
}

// ======================== Bottom Logs Panel ========================

fn render_bottom_logs_panel(ctx: &egui::Context, lab: &mut LabState) {
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
                            "PARAM_CHANGE" => egui::Color32::from_rgb(255, 200, 100),
                            "RUN_START" | "RUN_END" => egui::Color32::from_rgb(100, 255, 100),
                            "CONTROL" => egui::Color32::from_rgb(150, 200, 255),
                            "SCREENSHOT" | "SNAPSHOT" => egui::Color32::from_rgb(200, 150, 255),
                            _ => egui::Color32::from_rgb(180, 180, 180),
                        };
                        ui.label(egui::RichText::new(event.to_log_line()).small().color(color).monospace());
                    }
                });
        });
}

// ======================== Status Bar ========================

fn render_status_bar(ctx: &egui::Context, lab: &mut LabState) {
    if let Some(msg) = lab.current_status() {
        let msg = msg.to_string();
        egui::Area::new(egui::Id::new("status_bar"))
            .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(0.0, -10.0))
            .show(ctx, |ui| {
                egui::Frame::default()
                    .fill(egui::Color32::from_rgba_premultiplied(30, 80, 30, 220))
                    .corner_radius(egui::CornerRadius::same(4))
                    .inner_margin(egui::Margin::symmetric(12, 6))
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new(msg).color(egui::Color32::WHITE));
                    });
            });
    }
}

// ======================== Preset Save/Load ========================

fn save_preset(name: &str, params: &SimulationParams) {
    let dir = std::path::PathBuf::from("presets");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        log::error!("Failed to create presets dir: {}", e);
        return;
    }
    let path = dir.join(format!("{}.json", name));
    match serde_json::to_string_pretty(params) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&path, json) {
                log::error!("Failed to save preset: {}", e);
            } else {
                log::info!("Preset saved: {:?}", path);
            }
        }
        Err(e) => log::error!("Failed to serialize preset: {}", e),
    }
}

fn load_preset(name: &str) -> Option<SimulationParams> {
    let path = std::path::PathBuf::from(format!("presets/{}.json", name));
    let content = std::fs::read_to_string(&path).ok()?;
    match serde_json::from_str::<SimulationParams>(&content) {
        Ok(params) => {
            log::info!("Loaded preset from {:?}", path);
            Some(params)
        }
        Err(e) => {
            log::error!("Failed to parse preset {:?}: {}", path, e);
            None
        }
    }
}
