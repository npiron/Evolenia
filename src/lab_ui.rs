// ============================================================================
// lab_ui.rs — EvoLenia v2 Research Lab UI
// Complete egui-based interface for simulation control, parameter tuning,
// metrics visualization, experiment management, and data export.
// ============================================================================

use egui_plot::{Line, Plot, PlotPoints};

use crate::config::{
    visualization_mode_name, PerturbationType, SimulationParams, TIME_STEP_MAX, TIME_STEP_MIN,
    VIS_MODE_COUNT,
};
use crate::lab::LabState;
use crate::world::{target_total_mass, WORLD_HEIGHT, WORLD_WIDTH};

/// Main entry point for rendering all Research Lab UI panels.
pub fn render_lab_ui(ctx: &egui::Context, params: &mut SimulationParams, lab: &mut LabState) {
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

// ======================== Live Dashboard ========================

fn render_dashboard(ui: &mut egui::Ui, params: &mut SimulationParams, lab: &mut LabState) {
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
        .fill(egui::Color32::from_rgb(24, 28, 38))
        .corner_radius(8)
        .inner_margin(egui::Margin::symmetric(12, 10))
        .stroke(egui::Stroke::new(
            1.0,
            egui::Color32::from_rgb(60, 100, 160),
        ))
        .show(ui, |ui| {
            // Row 1: Key stats
            ui.horizontal(|ui| {
                dashboard_stat(
                    ui,
                    "🕐 Frame",
                    &format!("{}", frame),
                    egui::Color32::from_rgb(120, 220, 160),
                );
                ui.separator();
                dashboard_stat(
                    ui,
                    "⚡ FPS",
                    &format!("{:.0}", fps),
                    egui::Color32::from_rgb(255, 200, 100),
                );
                ui.separator();
                dashboard_stat(
                    ui,
                    "🧬 Sp.",
                    &format!("{}", species),
                    egui::Color32::from_rgb(200, 150, 255),
                );
            });
            ui.add_space(6.0);

            // Row 2: Gauges
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Vie")
                        .size(11.0)
                        .color(egui::Color32::from_rgb(140, 150, 170)),
                );
                ui.add(
                    egui::ProgressBar::new(live_frac / 100.0)
                        .desired_width(80.0)
                        .text(format!("{:.0}%", live_frac))
                        .fill(egui::Color32::from_rgb(100, 220, 140)),
                );
                ui.add_space(8.0);
                ui.label(
                    egui::RichText::new("Prédateurs")
                        .size(11.0)
                        .color(egui::Color32::from_rgb(140, 150, 170)),
                );
                ui.add(
                    egui::ProgressBar::new(predator_frac / 100.0)
                        .desired_width(80.0)
                        .text(format!("{:.0}%", predator_frac))
                        .fill(egui::Color32::from_rgb(255, 130, 100)),
                );
            });

            ui.add_space(8.0);
            ui.separator();
            ui.add_space(4.0);

            // Row 3: Play/Pause + Step + Speed + Restart
            ui.horizontal(|ui| {
                let (btn_text, btn_color) = if params.paused {
                    ("▶ Play", egui::Color32::from_rgb(60, 180, 80))
                } else {
                    ("⏸ Pause", egui::Color32::from_rgb(220, 150, 50))
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
                        .color(egui::Color32::from_rgb(160, 170, 190)),
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
                .color(egui::Color32::from_rgb(130, 140, 160)),
        );
        ui.label(egui::RichText::new(value).size(16.0).strong().color(color));
    });
}

// ======================== Minimal Overlay ========================

fn render_minimal_overlay(ctx: &egui::Context, params: &SimulationParams, lab: &mut LabState) {
    egui::Area::new(egui::Id::new("minimal_overlay"))
        .fixed_pos(egui::pos2(16.0, 16.0))
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(egui::Color32::from_rgb(18, 22, 32)) // Opaque background
                .corner_radius(10)
                .inner_margin(egui::Margin::symmetric(18, 14))
                .stroke(egui::Stroke::new(
                    2.0,
                    egui::Color32::from_rgb(60, 130, 200),
                ))
                .show(ui, |ui| {
                    let _pause_str = if params.paused { "  ⏸ PAUSED" } else { "" };
                    let frame = lab.metrics_history.last().map_or(0, |m| m.frame);
                    let fps = lab.metrics_history.last().map_or(0.0, |m| m.fps);

                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("🌌").size(24.0));
                        ui.add_space(8.0);
                        ui.label(
                            egui::RichText::new("EvoLenia")
                                .size(20.0)
                                .strong()
                                .color(egui::Color32::from_rgb(80, 180, 255)),
                        );
                        ui.add_space(20.0);
                        ui.label(
                            egui::RichText::new(format!("Frame: {}", frame))
                                .monospace()
                                .size(15.0)
                                .color(egui::Color32::from_rgb(120, 220, 160)),
                        );
                        ui.add_space(12.0);
                        ui.label(
                            egui::RichText::new("•")
                                .size(15.0)
                                .color(egui::Color32::from_rgb(80, 80, 100)),
                        );
                        ui.add_space(12.0);
                        ui.label(
                            egui::RichText::new(format!("FPS: {:.0}", fps))
                                .monospace()
                                .size(15.0)
                                .color(egui::Color32::from_rgb(255, 200, 100)),
                        );
                        if params.paused {
                            ui.add_space(12.0);
                            ui.label(
                                egui::RichText::new("⏸ PAUSED")
                                    .size(15.0)
                                    .strong()
                                    .color(egui::Color32::from_rgb(255, 120, 100)),
                            );
                        }
                    });

                    ui.add_space(6.0);
                    ui.label(
                        egui::RichText::new(
                            "F1 → Research Lab  •  Space → Pause  •  WASD → Pan  •  Q/E → Zoom",
                        )
                        .size(12.0)
                        .color(egui::Color32::from_rgb(130, 140, 160)),
                    );
                });
        });
}

// ======================== Left Panel ========================

fn render_left_panel(ctx: &egui::Context, params: &mut SimulationParams, lab: &mut LabState) {
    egui::SidePanel::left("lab_panel")
        .default_width(340.0)
        .min_width(300.0)
        .max_width(480.0)
        .show(ctx, |ui| {
            // ── HEADER ──
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("🔬 EvoLenia")
                        .size(20.0)
                        .strong()
                        .color(egui::Color32::from_rgb(100, 200, 255)),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .small_button("✕")
                        .on_hover_text("Hide Lab UI (F1)")
                        .clicked()
                    {
                        lab.show_lab_ui = false;
                    }
                });
            });
            ui.add_space(4.0);

            // ── LIVE DASHBOARD ──
            render_dashboard(ui, params, lab);
            ui.add_space(6.0);

            // ── SCROLLABLE ADVANCED SECTIONS ──
            egui::ScrollArea::vertical().show(ui, |ui| {
                render_control_section(ui, params, lab);
                ui.separator();
                render_visualization_section(ui, params);
                ui.separator();
                render_params_section(ui, params, lab);
                ui.separator();
                render_perturbation_section(ui, params, lab);
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

fn render_control_section(ui: &mut egui::Ui, params: &mut SimulationParams, lab: &mut LabState) {
    egui::CollapsingHeader::new("⏱ Time & Speed")
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Time Step:");
                if ui
                    .add(
                        egui::Slider::new(&mut params.time_step, TIME_STEP_MIN..=TIME_STEP_MAX)
                            .step_by(0.05)
                            .text("dt"),
                    )
                    .on_hover_text(
                        "Simulation time step — smaller = more stable but slower evolution",
                    )
                    .changed()
                {
                    lab.log_event(
                        0,
                        "PARAM_CHANGE",
                        &format!("time_step={:.2}", params.time_step),
                    );
                }
            });

            ui.horizontal(|ui| {
                ui.label("Analyse tous les:");
                ui.add(
                    egui::DragValue::new(&mut lab.metrics_sample_interval)
                        .range(10..=5000)
                        .suffix(" frames")
                        .speed(10),
                );
            })
            .response
            .on_hover_text("How often to read back metrics from GPU");

            ui.add_space(2.0);
            ui.label(
                egui::RichText::new(format!(
                    "dt effectif: {:.4}  ×  {} steps/frame",
                    0.1 * params.time_step,
                    params.simulation_speed,
                ))
                .small()
                .color(egui::Color32::from_rgb(150, 200, 150)),
            );
        });
}

// ======================== Parameters Section ========================

fn render_params_section(ui: &mut egui::Ui, params: &mut SimulationParams, lab: &mut LabState) {
    egui::CollapsingHeader::new("🧬 Paramètres avancés")
        .default_open(false)
        .show(ui, |ui| {
            ui.group(|ui| {
                ui.label(egui::RichText::new("Évolution / Mutation").strong());
                if ui.add(
                    egui::Slider::new(&mut params.mutation_rate, 0.1..=5.0)
                        .text("Taux de mutation")
                        .step_by(0.1),
                ).on_hover_text("Force du bruit génétique à chaque frame. Élevé = plus de diversification")
                    .changed() {
                    lab.log_event(0, "PARAM_CHANGE", &format!("mutation_rate={:.1}", params.mutation_rate));
                }
            });

            ui.group(|ui| {
                ui.label(egui::RichText::new("Prédation").strong());
                if ui.add(
                    egui::Slider::new(&mut params.predation_factor, 0.0..=3.0)
                        .text("Facteur de prédation")
                        .step_by(0.1),
                ).on_hover_text("Efficacité du transfert de masse lors d'une attaque. 0 = pas de prédation")
                    .changed() {
                    lab.log_event(0, "PARAM_CHANGE", &format!("predation={:.1}", params.predation_factor));
                }
            });

            ui.group(|ui| {
                ui.label(egui::RichText::new("Ressources (Gray-Scott)").strong());
                if ui.add(
                    egui::Slider::new(&mut params.resource_diffusion, 0.0..=0.5)
                        .text("Diffusion")
                        .step_by(0.01),
                ).on_hover_text("Vitesse de propagation des nutriments dans l'espace")
                    .changed() {
                    lab.log_event(0, "PARAM_CHANGE", &format!("diffusion={:.3}", params.resource_diffusion));
                }
                if ui.add(
                    egui::Slider::new(&mut params.resource_feed_rate, 0.0..=0.1)
                        .text("Apport")
                        .step_by(0.001),
                ).on_hover_text("Taux d'arrivée de nouveaux nutriments")
                    .changed() {
                    lab.log_event(0, "PARAM_CHANGE", &format!("feed_rate={:.4}", params.resource_feed_rate));
                }
                if ui.add(
                    egui::Slider::new(&mut params.resource_consumption, 0.0..=0.3)
                        .text("Consommation")
                        .step_by(0.01),
                ).on_hover_text("Quantité de nutriments consommés par les organismes")
                    .changed() {
                    lab.log_event(0, "PARAM_CHANGE", &format!("consumption={:.3}", params.resource_consumption));
                }
            });

            ui.group(|ui| {
                ui.label(egui::RichText::new("Normalisation de masse").strong());
                if ui.checkbox(&mut params.mass_normalization_enabled, "Activée")
                    .on_hover_text("Maintient la masse totale constante dans le monde")
                    .changed() {
                    lab.log_event(0, "PARAM_CHANGE", &format!("norm_enabled={}", params.mass_normalization_enabled));
                }
                if params.mass_normalization_enabled {
                    if ui.add(
                        egui::Slider::new(&mut params.mass_damping, 0.05..=1.0)
                            .text("Amortissement")
                            .step_by(0.05),
                    ).on_hover_text("Force du rappel vers la masse cible. 1.0 = correction totale immédiate")
                        .changed() {
                        lab.log_event(0, "PARAM_CHANGE", &format!("damping={:.2}", params.mass_damping));
                    }
                    if ui.add(
                        egui::Slider::new(&mut params.target_mass_multiplier, 0.1..=3.0)
                            .text("Masse cible ×")
                            .step_by(0.1),
                    ).on_hover_text("Multiplicateur de la masse totale cible")
                        .changed() {
                        lab.log_event(0, "PARAM_CHANGE", &format!("target_mass_mult={:.1}", params.target_mass_multiplier));
                    }
                    ui.label(
                        egui::RichText::new(format!(
                            "Cible: {:.0}",
                            target_total_mass() * params.target_mass_multiplier
                        ))
                        .small()
                        .color(egui::Color32::from_rgb(150, 200, 150)),
                    );
                }
            });

            ui.group(|ui| {
                ui.label(egui::RichText::new("⚖ Trade-offs").strong());
                if ui.add(
                    egui::Slider::new(&mut params.radius_cost_exponent, 1.0..=3.0)
                        .text("Coût du radius")
                        .step_by(0.1),
                ).on_hover_text("Pénalité exponentielle pour les grands rayons de perception")
                    .changed() {
                    lab.log_event(0, "PARAM_CHANGE", &format!("radius_cost_exp={:.1}", params.radius_cost_exponent));
                }
                if ui.add(
                    egui::Slider::new(&mut params.agg_mobility_tradeoff, 0.0..=1.0)
                        .text("Aggressivité ↔ Mobilité")
                        .step_by(0.05),
                ).on_hover_text("Plus un organisme est agressif, moins il se déplace vite")
                    .changed() {
                    lab.log_event(0, "PARAM_CHANGE", &format!("agg_mobility={:.2}", params.agg_mobility_tradeoff));
                }
                if ui.add(
                    egui::Slider::new(&mut params.starvation_severity, 0.01..=0.2)
                        .text("Sévérité famine")
                        .step_by(0.005),
                ).on_hover_text("Pénalité énergétique quand les ressources sont basses")
                    .changed() {
                    lab.log_event(0, "PARAM_CHANGE", &format!("starvation={:.3}", params.starvation_severity));
                }
            });

            ui.group(|ui| {
                ui.label(egui::RichText::new("Conditions initiales (au restart)").strong());
                ui.add(
                    egui::Slider::new(&mut params.num_seed_clusters, 5..=100)
                        .text("Clusters de départ"),
                ).on_hover_text("Nombre de groupes d'organismes initiaux");
                ui.add(
                    egui::Slider::new(&mut params.seed_cluster_size, 0.5..=3.0)
                        .text("Taille des clusters")
                        .step_by(0.1),
                );
                ui.add(
                    egui::Slider::new(&mut params.initial_mass_fill, 0.05..=0.5)
                        .text("Remplissage initial %")
                        .step_by(0.01),
                ).on_hover_text("Pourcentage de la masse cible au démarrage");
            });
        });
}

// ======================== Perturbation Section ========================

fn render_perturbation_section(
    ui: &mut egui::Ui,
    params: &mut SimulationParams,
    lab: &mut LabState,
) {
    egui::CollapsingHeader::new("🌊 Perturbations")
        .default_open(false)
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new("Appliquer des perturbations écologiques à la simulation")
                    .small()
                    .color(egui::Color32::GRAY),
            );

            egui::ComboBox::from_label("Type")
                .selected_text(params.perturbation_type.name())
                .show_ui(ui, |ui| {
                    for pt in PerturbationType::all() {
                        ui.selectable_value(&mut params.perturbation_type, pt.clone(), pt.name());
                    }
                });

            ui.add(
                egui::Slider::new(&mut params.perturbation_intensity, 0.0..=1.0)
                    .text("Intensité")
                    .step_by(0.05),
            )
            .on_hover_text("Force de la perturbation");

            ui.add(
                egui::Slider::new(&mut params.perturbation_radius, 0.05..=0.5)
                    .text("Rayon")
                    .step_by(0.01),
            )
            .on_hover_text("Taille de la zone affectée (fraction du monde)");

            ui.horizontal(|ui| {
                ui.label("Centre:");
                ui.add(
                    egui::DragValue::new(&mut params.perturbation_center_x)
                        .range(0.0..=1.0)
                        .speed(0.01)
                        .prefix("x="),
                );
                ui.add(
                    egui::DragValue::new(&mut params.perturbation_center_y)
                        .range(0.0..=1.0)
                        .speed(0.01)
                        .prefix("y="),
                );
            });

            ui.add_space(4.0);

            let desc = match params.perturbation_type {
                PerturbationType::None => "Aucune perturbation sélectionnée",
                PerturbationType::Drought => "Détruit les ressources dans la zone",
                PerturbationType::NutrientPulse => "Injecte des ressources supplémentaires",
                PerturbationType::MassStorm => "Tue les organismes (masse → 0)",
                PerturbationType::MutationBurst => "Randomise l'ADN dans la zone",
            };
            ui.label(
                egui::RichText::new(desc)
                    .small()
                    .italics()
                    .color(egui::Color32::from_rgb(200, 200, 150)),
            );

            ui.add_space(4.0);

            let can_apply = params.perturbation_type != PerturbationType::None;
            ui.add_enabled_ui(can_apply, |ui| {
                if ui.button("⚡ Appliquer").clicked() {
                    params.perturbation_active = true;
                    lab.log_event(
                        0,
                        "PERTURBATION",
                        &format!(
                            "{} intensity={:.2} radius={:.2}",
                            params.perturbation_type.name(),
                            params.perturbation_intensity,
                            params.perturbation_radius,
                        ),
                    );
                }
            });

            if params.perturbation_active {
                ui.label(
                    egui::RichText::new("● En attente…")
                        .color(egui::Color32::from_rgb(255, 200, 50)),
                );
            }
        });
}

// ======================== Visualization Section ========================

fn render_visualization_section(ui: &mut egui::Ui, params: &mut SimulationParams) {
    egui::CollapsingHeader::new("🎨 Visualisation")
        .default_open(true)
        .show(ui, |ui| {
            for mode in 0..VIS_MODE_COUNT {
                let name = visualization_mode_name(mode);
                if ui
                    .radio_value(&mut params.visualization_mode, mode, name)
                    .clicked()
                {
                    log::info!("Visualization mode: {}", name);
                }
            }
            ui.add_space(4.0);
            ui.checkbox(&mut params.vsync, "VSync").on_hover_text(
                "Active la synchronisation verticale (limite FPS au rafraîchissement écran)",
            );

            ui.label(
                egui::RichText::new(format!("Monde: {}×{}", WORLD_WIDTH, WORLD_HEIGHT))
                    .small()
                    .color(egui::Color32::GRAY),
            );
        });
}

// ======================== Experiment Section ========================

fn render_experiment_section(ui: &mut egui::Ui, params: &mut SimulationParams, lab: &mut LabState) {
    egui::CollapsingHeader::new("🧪 Expériences")
        .default_open(false)
        .show(ui, |ui| {
            // Seed control
            ui.group(|ui| {
                ui.label(egui::RichText::new("Reproducibility").strong());
                ui.checkbox(&mut params.use_fixed_seed, "Use fixed seed");
                if params.use_fixed_seed {
                    ui.horizontal(|ui| {
                        ui.label("Seed:");
                        ui.add(
                            egui::DragValue::new(&mut params.fixed_seed_value).range(0..=u64::MAX),
                        );
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

            // Presets - Full Selector System
            ui.group(|ui| {
                ui.label(
                    egui::RichText::new("🎛️ Preset Selector")
                        .strong()
                        .size(14.0),
                );
                ui.add_space(4.0);

                let catalog = preset_catalog();
                let categories = preset_categories();

                // Category tabs
                ui.horizontal_wrapped(|ui| {
                    for (cat_idx, cat) in categories.iter().enumerate() {
                        let cat_emoji = match *cat {
                            "Autonomous" => "🛸",
                            "Predation" => "🦁",
                            "Explosive" => "💥",
                            "Stable" => "🪷",
                            "Special" => "✨",
                            "Experimental" => "🧬",
                            _ => "📁",
                        };

                        // Count presets in this category
                        let first_in_cat =
                            catalog.iter().position(|p| p.category == *cat).unwrap_or(0);

                        let is_selected = catalog
                            .get(lab.selected_preset_index)
                            .map(|p| p.category == *cat)
                            .unwrap_or(cat_idx == 0);

                        let btn = egui::Button::new(format!("{} {}", cat_emoji, cat)).fill(
                            if is_selected {
                                egui::Color32::from_rgb(60, 100, 160)
                            } else {
                                egui::Color32::from_rgb(40, 45, 55)
                            },
                        );

                        if ui.add(btn).clicked() {
                            lab.selected_preset_index = first_in_cat;
                        }
                    }
                });

                ui.add_space(6.0);
                ui.separator();
                ui.add_space(4.0);

                // Preset list for selected category
                let current_cat = catalog
                    .get(lab.selected_preset_index)
                    .map(|p| p.category)
                    .unwrap_or("Autonomous");

                let cat_presets: Vec<(usize, &PresetInfo)> = catalog
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.category == current_cat)
                    .collect();

                egui::ScrollArea::vertical()
                    .max_height(180.0)
                    .show(ui, |ui| {
                        for (idx, preset) in cat_presets {
                            let is_selected = idx == lab.selected_preset_index;

                            ui.horizontal(|ui| {
                                let btn_color = if is_selected {
                                    egui::Color32::from_rgb(80, 140, 200)
                                } else {
                                    egui::Color32::from_rgb(50, 55, 65)
                                };

                                let btn =
                                    egui::Button::new(format!("{} {}", preset.emoji, preset.name))
                                        .fill(btn_color)
                                        .min_size(egui::vec2(140.0, 24.0));

                                if ui.add(btn).clicked() {
                                    lab.selected_preset_index = idx;
                                }
                            });

                            if is_selected {
                                ui.indent("desc", |ui| {
                                    ui.label(
                                        egui::RichText::new(preset.description)
                                            .small()
                                            .color(egui::Color32::from_rgb(180, 180, 200)),
                                    );
                                });
                                ui.add_space(4.0);
                            }
                        }
                    });

                ui.add_space(6.0);
                ui.separator();
                ui.add_space(4.0);

                // Action buttons
                ui.horizontal(|ui| {
                    if ui.button("▶️ Apply & Restart").clicked() {
                        if let Some(preset) = catalog.get(lab.selected_preset_index) {
                            let vis = params.visualization_mode; // Keep current view
                            *params = (preset.params)();
                            params.visualization_mode = vis;
                            lab.restart_requested = true;
                            lab.set_status(format!("{} {} applied!", preset.emoji, preset.name));
                        }
                    }

                    if ui.button("📋 Apply Only").clicked() {
                        if let Some(preset) = catalog.get(lab.selected_preset_index) {
                            let vis = params.visualization_mode;
                            *params = (preset.params)();
                            params.visualization_mode = vis;
                            lab.set_status(format!(
                                "{} {} params set (no restart)",
                                preset.emoji, preset.name
                            ));
                        }
                    }
                });

                ui.add_space(4.0);

                // Custom preset save/load
                ui.collapsing("💾 Custom Presets", |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Name:");
                        ui.text_edit_singleline(&mut lab.preset_name);
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Save").clicked() {
                            save_preset(&lab.preset_name, params);
                            lab.set_status(format!("Preset '{}' saved to disk", lab.preset_name));
                        }
                        if ui.button("Load").clicked() {
                            if let Some(loaded) = load_preset(&lab.preset_name) {
                                *params = loaded;
                                lab.set_status(format!("Preset '{}' loaded", lab.preset_name));
                            } else {
                                lab.set_status(format!("Preset '{}' not found", lab.preset_name));
                            }
                        }
                    });
                });

                if ui.button("🔄 Reset to defaults").clicked() {
                    let vis = params.visualization_mode;
                    *params = SimulationParams::default();
                    params.visualization_mode = vis;
                    lab.set_status("Parameters reset to defaults".to_string());
                }
            });
        });
}

// ======================== Capture Section ========================

fn render_capture_section(ui: &mut egui::Ui, params: &SimulationParams, lab: &mut LabState) {
    egui::CollapsingHeader::new("📸 Capture & Export")
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("📷 Screenshot (F12)").clicked() {
                    lab.screenshot_requested = true;
                }
                if ui.button("💾 Snapshot").clicked() {
                    lab.snapshot_requested = true;
                }
            })
            .response
            .on_hover_text(
                "Screenshot = image PNG | Snapshot = état complet de la simulation (.snap)",
            );

            if ui.button("📊 Export Metrics CSV").clicked() {
                match lab.export_metrics_csv() {
                    Ok(path) => lab.set_status(format!("Exporté: {:?}", path)),
                    Err(e) => lab.set_status(format!("Échec export: {}", e)),
                }
            }

            if ui.button("📝 Export Rapport").clicked() {
                match lab.export_report(params) {
                    Ok(path) => lab.set_status(format!("Rapport: {:?}", path)),
                    Err(e) => lab.set_status(format!("Échec rapport: {}", e)),
                }
            }
        });
}

// ======================== View Toggles ========================

fn render_view_toggles(ui: &mut egui::Ui, lab: &mut LabState) {
    egui::CollapsingHeader::new("📊 Panneaux")
        .default_open(false)
        .show(ui, |ui| {
            ui.checkbox(&mut lab.show_analysis_panel, "Panneau d'analyse (F9)")
                .on_hover_text("Affiche les graphiques temps réel (entropie, espèces, FPS...)");
            ui.checkbox(&mut lab.show_logs_panel, "Journal d'événements")
                .on_hover_text("Historique des actions (pause, restart, perturbations...)");
        });
}

// ======================== Right Analysis Panel ========================

fn render_right_analysis_panel(ctx: &egui::Context, lab: &mut LabState) {
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
                        .color(egui::Color32::from_rgb(150, 220, 150)),
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
                        stat_row(
                            ui,
                            "Live Pixels",
                            &format!("{} ({:.1}%)", last.live_pixels, last.live_fraction * 100.0),
                        );
                        stat_row(
                            ui,
                            "Predators",
                            &format!("{:.1}%", last.predator_fraction * 100.0),
                        );
                        stat_row(ui, "Avg Resource", &format!("{:.3}", last.avg_resource));
                        stat_row(ui, "Mass StdDev", &format!("{:.4}", last.mass_std_dev));
                        // Phase 1 eco metrics
                        stat_row(ui, "Prey %", &format!("{:.1}%", last.prey_fraction * 100.0));
                        stat_row(
                            ui,
                            "Opportunist %",
                            &format!("{:.1}%", last.opportunist_fraction * 100.0),
                        );
                        stat_row(
                            ui,
                            "Eff. Diversity",
                            &format!("{:.2}", last.effective_diversity),
                        );
                        stat_row(ui, "Genome Var", &format!("{:.4}", last.genome_variance));
                        stat_row(ui, "Total Energy", &format!("{:.0}", last.total_energy));
                        stat_row(ui, "Energy Flux", &format!("{:.4}", last.energy_flux));
                    });
            }
            ui.separator();

            // Time-series plots
            egui::ScrollArea::vertical().show(ui, |ui| {
                render_plot(ui, "Total Mass", &lab.metrics_history, |m| {
                    m.total_mass as f64
                });
                render_plot(ui, "Avg Energy", &lab.metrics_history, |m| {
                    m.avg_energy as f64
                });
                render_plot(ui, "Genetic Entropy", &lab.metrics_history, |m| {
                    m.entropy as f64
                });
                render_plot(ui, "Species Count", &lab.metrics_history, |m| {
                    m.species as f64
                });
                render_plot(ui, "Live Pixels", &lab.metrics_history, |m| {
                    m.live_pixels as f64
                });
                render_plot(ui, "FPS", &lab.metrics_history, |m| m.fps as f64);

                // Phase 1 eco plots
                render_plot(ui, "Effective Diversity", &lab.metrics_history, |m| {
                    m.effective_diversity as f64
                });
                render_plot(ui, "Energy Flux", &lab.metrics_history, |m| {
                    m.energy_flux as f64
                });
                render_plot(ui, "Genome Variance", &lab.metrics_history, |m| {
                    m.genome_variance as f64
                });

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
    ui.label(egui::RichText::new(label).color(egui::Color32::from_rgb(180, 180, 200)));
    ui.label(
        egui::RichText::new(value)
            .monospace()
            .strong()
            .color(egui::Color32::from_rgb(220, 220, 240)),
    );
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

// ======================== Preset Catalog ========================

/// A preset with name, description, and parameters
pub struct PresetInfo {
    pub name: &'static str,
    pub emoji: &'static str,
    pub description: &'static str,
    pub category: &'static str,
    pub params: fn() -> SimulationParams,
}

/// Built-in preset catalog - all the interesting configurations
/// Each preset defines ALL 29 parameters explicitly for complete control.
pub fn preset_catalog() -> Vec<PresetInfo> {
    vec![
        // ============ AUTONOMY & MOVEMENT ============
        PresetInfo {
            name: "Lenia Gliders",
            emoji: "🛸",
            description: "Classic Lenia creatures (orbium, geminium). Self-propelled organisms that glide gracefully across the world.",
            category: "Autonomous",
            params: || SimulationParams {
                // Control
                paused: false,
                simulation_speed: 1,
                time_step: 0.8,
                vsync: false,
                // Visualization
                visualization_mode: 0,
                show_extended_ui: false,
                // Evolution
                mutation_rate: 0.15,
                predation_factor: 0.5,
                // Resources
                resource_diffusion: 0.08,
                resource_feed_rate: 0.012,
                resource_consumption: 0.045,
                // Mass
                mass_normalization_enabled: true,
                mass_damping: 0.4,
                target_mass_multiplier: 0.85,
                // Trade-offs
                radius_cost_exponent: 1.0,
                agg_mobility_tradeoff: 0.05,
                starvation_severity: 0.01,
                // Perturbations
                perturbation_type: PerturbationType::None,
                perturbation_intensity: 0.5,
                perturbation_radius: 0.15,
                perturbation_active: false,
                perturbation_center_x: 0.5,
                perturbation_center_y: 0.5,
                // Initial conditions
                num_seed_clusters: 25,
                seed_cluster_size: 1.8,
                initial_mass_fill: 0.22,
                // Reproducibility
                seed: None,
                use_fixed_seed: false,
                fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Swarm Drones",
            emoji: "🐝",
            description: "Fast-moving swarms of small entities. High mobility, coordinated movements, emergent flocking behavior.",
            category: "Autonomous",
            params: || SimulationParams {
                paused: false,
                simulation_speed: 1,
                time_step: 1.0,
                vsync: false,
                visualization_mode: 0,
                show_extended_ui: false,
                mutation_rate: 0.8,
                predation_factor: 1.5,
                resource_diffusion: 0.12,
                resource_feed_rate: 0.018,
                resource_consumption: 0.04,
                mass_normalization_enabled: false,
                mass_damping: 0.15,
                target_mass_multiplier: 1.2,
                radius_cost_exponent: 1.1,
                agg_mobility_tradeoff: 0.15,
                starvation_severity: 0.02,
                perturbation_type: PerturbationType::None,
                perturbation_intensity: 0.5,
                perturbation_radius: 0.15,
                perturbation_active: false,
                perturbation_center_x: 0.5,
                perturbation_center_y: 0.5,
                num_seed_clusters: 50,
                seed_cluster_size: 0.8,
                initial_mass_fill: 0.12,
                seed: None,
                use_fixed_seed: false,
                fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Wandering Blobs",
            emoji: "🫧",
            description: "Large slow-moving organisms with low mutation. They drift and pulse like jellyfish.",
            category: "Autonomous",
            params: || SimulationParams {
                paused: false,
                simulation_speed: 1,
                time_step: 0.6,
                vsync: false,
                visualization_mode: 0,
                show_extended_ui: false,
                mutation_rate: 0.1,
                predation_factor: 0.3,
                resource_diffusion: 0.06,
                resource_feed_rate: 0.014,
                resource_consumption: 0.035,
                mass_normalization_enabled: true,
                mass_damping: 0.5,
                target_mass_multiplier: 0.7,
                radius_cost_exponent: 0.9,
                agg_mobility_tradeoff: 0.0,
                starvation_severity: 0.005,
                perturbation_type: PerturbationType::None,
                perturbation_intensity: 0.5,
                perturbation_radius: 0.15,
                perturbation_active: false,
                perturbation_center_x: 0.5,
                perturbation_center_y: 0.5,
                num_seed_clusters: 15,
                seed_cluster_size: 2.5,
                initial_mass_fill: 0.18,
                seed: None,
                use_fixed_seed: false,
                fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Micro Rockets",
            emoji: "🚀",
            description: "Tiny fast entities that shoot across the screen. High speed, rapid evolution.",
            category: "Autonomous",
            params: || SimulationParams {
                paused: false,
                simulation_speed: 1,
                time_step: 1.2,
                vsync: false,
                visualization_mode: 0,
                show_extended_ui: false,
                mutation_rate: 1.5,
                predation_factor: 0.8,
                resource_diffusion: 0.15,
                resource_feed_rate: 0.022,
                resource_consumption: 0.03,
                mass_normalization_enabled: false,
                mass_damping: 0.1,
                target_mass_multiplier: 1.5,
                radius_cost_exponent: 0.8,
                agg_mobility_tradeoff: 0.1,
                starvation_severity: 0.015,
                perturbation_type: PerturbationType::None,
                perturbation_intensity: 0.5,
                perturbation_radius: 0.15,
                perturbation_active: false,
                perturbation_center_x: 0.5,
                perturbation_center_y: 0.5,
                num_seed_clusters: 80,
                seed_cluster_size: 0.5,
                initial_mass_fill: 0.08,
                seed: None,
                use_fixed_seed: false,
                fixed_seed_value: 42,
            },
        },

        // ============ PREDATION & COMPETITION ============
        PresetInfo {
            name: "Predator Prey",
            emoji: "🦁",
            description: "Intense hunter-prey dynamics. Predators chase, prey flee. Watch the ecosystem balance!",
            category: "Predation",
            params: || SimulationParams {
                paused: false,
                simulation_speed: 1,
                time_step: 1.0,
                vsync: false,
                visualization_mode: 0,
                show_extended_ui: false,
                mutation_rate: 0.5,
                predation_factor: 2.5,
                resource_diffusion: 0.07,
                resource_feed_rate: 0.016,
                resource_consumption: 0.07,
                mass_normalization_enabled: true,
                mass_damping: 0.3,
                target_mass_multiplier: 1.0,
                radius_cost_exponent: 1.4,
                agg_mobility_tradeoff: 0.6,
                starvation_severity: 0.04,
                perturbation_type: PerturbationType::None,
                perturbation_intensity: 0.5,
                perturbation_radius: 0.15,
                perturbation_active: false,
                perturbation_center_x: 0.5,
                perturbation_center_y: 0.5,
                num_seed_clusters: 35,
                seed_cluster_size: 1.2,
                initial_mass_fill: 0.20,
                seed: None,
                use_fixed_seed: false,
                fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Arms Race",
            emoji: "⚔️",
            description: "High mutation + high predation = constant evolutionary warfare. Species rise and fall.",
            category: "Predation",
            params: || SimulationParams {
                paused: false,
                simulation_speed: 1,
                time_step: 1.1,
                vsync: false,
                visualization_mode: 0,
                show_extended_ui: false,
                mutation_rate: 2.0,
                predation_factor: 3.0,
                resource_diffusion: 0.08,
                resource_feed_rate: 0.020,
                resource_consumption: 0.08,
                mass_normalization_enabled: true,
                mass_damping: 0.25,
                target_mass_multiplier: 1.1,
                radius_cost_exponent: 1.5,
                agg_mobility_tradeoff: 0.7,
                starvation_severity: 0.06,
                perturbation_type: PerturbationType::None,
                perturbation_intensity: 0.5,
                perturbation_radius: 0.15,
                perturbation_active: false,
                perturbation_center_x: 0.5,
                perturbation_center_y: 0.5,
                num_seed_clusters: 40,
                seed_cluster_size: 1.0,
                initial_mass_fill: 0.18,
                seed: None,
                use_fixed_seed: false,
                fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Apex Hunters",
            emoji: "🦈",
            description: "Few powerful predators dominate. Sparse resources, harsh survival.",
            category: "Predation",
            params: || SimulationParams {
                paused: false,
                simulation_speed: 1,
                time_step: 0.9,
                vsync: false,
                visualization_mode: 0,
                show_extended_ui: false,
                mutation_rate: 0.3,
                predation_factor: 4.0,
                resource_diffusion: 0.05,
                resource_feed_rate: 0.008,
                resource_consumption: 0.10,
                mass_normalization_enabled: true,
                mass_damping: 0.4,
                target_mass_multiplier: 0.8,
                radius_cost_exponent: 1.6,
                agg_mobility_tradeoff: 0.8,
                starvation_severity: 0.08,
                perturbation_type: PerturbationType::None,
                perturbation_intensity: 0.5,
                perturbation_radius: 0.15,
                perturbation_active: false,
                perturbation_center_x: 0.5,
                perturbation_center_y: 0.5,
                num_seed_clusters: 20,
                seed_cluster_size: 1.5,
                initial_mass_fill: 0.15,
                seed: None,
                use_fixed_seed: false,
                fixed_seed_value: 42,
            },
        },

        // ============ EXPLOSIVE & CHAOTIC ============
        PresetInfo {
            name: "Big Bang",
            emoji: "💥",
            description: "Explosive growth from few seeds. Watch life rapidly colonize the entire world!",
            category: "Explosive",
            params: || SimulationParams {
                paused: false,
                simulation_speed: 1,
                time_step: 1.3,
                vsync: false,
                visualization_mode: 0,
                show_extended_ui: false,
                mutation_rate: 2.5,
                predation_factor: 1.0,
                resource_diffusion: 0.18,
                resource_feed_rate: 0.030,
                resource_consumption: 0.025,
                mass_normalization_enabled: false,
                mass_damping: 0.05,
                target_mass_multiplier: 2.0,
                radius_cost_exponent: 0.7,
                agg_mobility_tradeoff: 0.1,
                starvation_severity: 0.008,
                perturbation_type: PerturbationType::None,
                perturbation_intensity: 0.5,
                perturbation_radius: 0.15,
                perturbation_active: false,
                perturbation_center_x: 0.5,
                perturbation_center_y: 0.5,
                num_seed_clusters: 5,
                seed_cluster_size: 3.0,
                initial_mass_fill: 0.05,
                seed: None,
                use_fixed_seed: false,
                fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Chaos Storm",
            emoji: "🌪️",
            description: "Maximum entropy! Wild mutations, unstable dynamics, kaleidoscopic patterns.",
            category: "Explosive",
            params: || SimulationParams {
                paused: false,
                simulation_speed: 2,
                time_step: 1.5,
                vsync: false,
                visualization_mode: 0,
                show_extended_ui: false,
                mutation_rate: 5.0,
                predation_factor: 2.0,
                resource_diffusion: 0.20,
                resource_feed_rate: 0.025,
                resource_consumption: 0.05,
                mass_normalization_enabled: false,
                mass_damping: 0.02,
                target_mass_multiplier: 1.8,
                radius_cost_exponent: 0.6,
                agg_mobility_tradeoff: 0.3,
                starvation_severity: 0.01,
                perturbation_type: PerturbationType::None,
                perturbation_intensity: 0.5,
                perturbation_radius: 0.15,
                perturbation_active: false,
                perturbation_center_x: 0.5,
                perturbation_center_y: 0.5,
                num_seed_clusters: 60,
                seed_cluster_size: 0.7,
                initial_mass_fill: 0.15,
                seed: None,
                use_fixed_seed: false,
                fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Nova Burst",
            emoji: "⭐",
            description: "Periodic explosions of life followed by collapses. Boom-bust cycles.",
            category: "Explosive",
            params: || SimulationParams {
                paused: false,
                simulation_speed: 1,
                time_step: 1.2,
                vsync: false,
                visualization_mode: 0,
                show_extended_ui: false,
                mutation_rate: 1.8,
                predation_factor: 1.5,
                resource_diffusion: 0.10,
                resource_feed_rate: 0.035,
                resource_consumption: 0.06,
                mass_normalization_enabled: true,
                mass_damping: 0.2,
                target_mass_multiplier: 1.3,
                radius_cost_exponent: 1.0,
                agg_mobility_tradeoff: 0.2,
                starvation_severity: 0.03,
                perturbation_type: PerturbationType::None,
                perturbation_intensity: 0.5,
                perturbation_radius: 0.15,
                perturbation_active: false,
                perturbation_center_x: 0.5,
                perturbation_center_y: 0.5,
                num_seed_clusters: 30,
                seed_cluster_size: 1.2,
                initial_mass_fill: 0.12,
                seed: None,
                use_fixed_seed: false,
                fixed_seed_value: 42,
            },
        },

        // ============ STABLE & AESTHETIC ============
        PresetInfo {
            name: "Zen Garden",
            emoji: "🪷",
            description: "Peaceful, stable patterns. Slow evolution, beautiful organic shapes.",
            category: "Stable",
            params: || SimulationParams {
                paused: false,
                simulation_speed: 1,
                time_step: 0.5,
                vsync: false,
                visualization_mode: 0,
                show_extended_ui: false,
                mutation_rate: 0.05,
                predation_factor: 0.2,
                resource_diffusion: 0.04,
                resource_feed_rate: 0.010,
                resource_consumption: 0.03,
                mass_normalization_enabled: true,
                mass_damping: 0.6,
                target_mass_multiplier: 0.6,
                radius_cost_exponent: 1.2,
                agg_mobility_tradeoff: 0.0,
                starvation_severity: 0.003,
                perturbation_type: PerturbationType::None,
                perturbation_intensity: 0.5,
                perturbation_radius: 0.15,
                perturbation_active: false,
                perturbation_center_x: 0.5,
                perturbation_center_y: 0.5,
                num_seed_clusters: 40,
                seed_cluster_size: 1.5,
                initial_mass_fill: 0.25,
                seed: None,
                use_fixed_seed: false,
                fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Coral Reef",
            emoji: "🪸",
            description: "Dense, colorful colonies. Slow-growing structures with high biodiversity.",
            category: "Stable",
            params: || SimulationParams {
                paused: false,
                simulation_speed: 1,
                time_step: 0.7,
                vsync: false,
                visualization_mode: 0,
                show_extended_ui: false,
                mutation_rate: 0.2,
                predation_factor: 0.4,
                resource_diffusion: 0.06,
                resource_feed_rate: 0.014,
                resource_consumption: 0.04,
                mass_normalization_enabled: true,
                mass_damping: 0.45,
                target_mass_multiplier: 0.75,
                radius_cost_exponent: 1.1,
                agg_mobility_tradeoff: 0.1,
                starvation_severity: 0.01,
                perturbation_type: PerturbationType::None,
                perturbation_intensity: 0.5,
                perturbation_radius: 0.15,
                perturbation_active: false,
                perturbation_center_x: 0.5,
                perturbation_center_y: 0.5,
                num_seed_clusters: 70,
                seed_cluster_size: 0.8,
                initial_mass_fill: 0.28,
                seed: None,
                use_fixed_seed: false,
                fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Crystal Growth",
            emoji: "💎",
            description: "Geometric, ordered growth patterns. Low chaos, high structure.",
            category: "Stable",
            params: || SimulationParams {
                paused: false,
                simulation_speed: 1,
                time_step: 0.4,
                vsync: false,
                visualization_mode: 0,
                show_extended_ui: false,
                mutation_rate: 0.02,
                predation_factor: 0.1,
                resource_diffusion: 0.03,
                resource_feed_rate: 0.008,
                resource_consumption: 0.025,
                mass_normalization_enabled: true,
                mass_damping: 0.7,
                target_mass_multiplier: 0.5,
                radius_cost_exponent: 1.3,
                agg_mobility_tradeoff: 0.0,
                starvation_severity: 0.002,
                perturbation_type: PerturbationType::None,
                perturbation_intensity: 0.5,
                perturbation_radius: 0.15,
                perturbation_active: false,
                perturbation_center_x: 0.5,
                perturbation_center_y: 0.5,
                num_seed_clusters: 30,
                seed_cluster_size: 2.0,
                initial_mass_fill: 0.30,
                seed: None,
                use_fixed_seed: false,
                fixed_seed_value: 42,
            },
        },

        // ============ SPECIAL EFFECTS ============
        PresetInfo {
            name: "Fireflies",
            emoji: "🔥",
            description: "Sparse glowing entities that flicker and dance in the dark.",
            category: "Special",
            params: || SimulationParams {
                paused: false,
                simulation_speed: 1,
                time_step: 0.9,
                vsync: false,
                visualization_mode: 0,
                show_extended_ui: false,
                mutation_rate: 0.6,
                predation_factor: 0.6,
                resource_diffusion: 0.09,
                resource_feed_rate: 0.011,
                resource_consumption: 0.055,
                mass_normalization_enabled: true,
                mass_damping: 0.35,
                target_mass_multiplier: 0.9,
                radius_cost_exponent: 1.0,
                agg_mobility_tradeoff: 0.2,
                starvation_severity: 0.02,
                perturbation_type: PerturbationType::None,
                perturbation_intensity: 0.5,
                perturbation_radius: 0.15,
                perturbation_active: false,
                perturbation_center_x: 0.5,
                perturbation_center_y: 0.5,
                num_seed_clusters: 100,
                seed_cluster_size: 0.4,
                initial_mass_fill: 0.06,
                seed: None,
                use_fixed_seed: false,
                fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Lava Flow",
            emoji: "🌋",
            description: "Slow-spreading mass that flows and pools. Use Thermal visualization mode!",
            category: "Special",
            params: || SimulationParams {
                paused: false,
                simulation_speed: 1,
                time_step: 0.6,
                vsync: false,
                visualization_mode: 2, // Thermal mode
                show_extended_ui: false,
                mutation_rate: 0.3,
                predation_factor: 0.7,
                resource_diffusion: 0.15,
                resource_feed_rate: 0.020,
                resource_consumption: 0.02,
                mass_normalization_enabled: false,
                mass_damping: 0.08,
                target_mass_multiplier: 1.6,
                radius_cost_exponent: 0.8,
                agg_mobility_tradeoff: 0.05,
                starvation_severity: 0.005,
                perturbation_type: PerturbationType::None,
                perturbation_intensity: 0.5,
                perturbation_radius: 0.15,
                perturbation_active: false,
                perturbation_center_x: 0.5,
                perturbation_center_y: 0.5,
                num_seed_clusters: 10,
                seed_cluster_size: 3.5,
                initial_mass_fill: 0.10,
                seed: None,
                use_fixed_seed: false,
                fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Neural Network",
            emoji: "🧠",
            description: "Interconnected pathways that form and dissolve. Brain-like connectivity.",
            category: "Special",
            params: || SimulationParams {
                paused: false,
                simulation_speed: 1,
                time_step: 0.85,
                vsync: false,
                visualization_mode: 0,
                show_extended_ui: false,
                mutation_rate: 0.4,
                predation_factor: 0.5,
                resource_diffusion: 0.07,
                resource_feed_rate: 0.013,
                resource_consumption: 0.05,
                mass_normalization_enabled: true,
                mass_damping: 0.38,
                target_mass_multiplier: 0.85,
                radius_cost_exponent: 1.15,
                agg_mobility_tradeoff: 0.15,
                starvation_severity: 0.015,
                perturbation_type: PerturbationType::None,
                perturbation_intensity: 0.5,
                perturbation_radius: 0.15,
                perturbation_active: false,
                perturbation_center_x: 0.5,
                perturbation_center_y: 0.5,
                num_seed_clusters: 45,
                seed_cluster_size: 1.1,
                initial_mass_fill: 0.20,
                seed: None,
                use_fixed_seed: false,
                fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Galaxy Formation",
            emoji: "🌌",
            description: "Spiral arms emerge from rotating mass concentrations. Cosmic scale!",
            category: "Special",
            params: || SimulationParams {
                paused: false,
                simulation_speed: 1,
                time_step: 0.75,
                vsync: false,
                visualization_mode: 0,
                show_extended_ui: false,
                mutation_rate: 0.25,
                predation_factor: 0.8,
                resource_diffusion: 0.11,
                resource_feed_rate: 0.016,
                resource_consumption: 0.038,
                mass_normalization_enabled: true,
                mass_damping: 0.3,
                target_mass_multiplier: 1.0,
                radius_cost_exponent: 1.05,
                agg_mobility_tradeoff: 0.25,
                starvation_severity: 0.012,
                perturbation_type: PerturbationType::None,
                perturbation_intensity: 0.5,
                perturbation_radius: 0.15,
                perturbation_active: false,
                perturbation_center_x: 0.5,
                perturbation_center_y: 0.5,
                num_seed_clusters: 8,
                seed_cluster_size: 4.0,
                initial_mass_fill: 0.08,
                seed: None,
                use_fixed_seed: false,
                fixed_seed_value: 42,
            },
        },

        // ============ EXPERIMENTAL ============
        PresetInfo {
            name: "Evolution Lab",
            emoji: "🧬",
            description: "Balanced parameters for observing long-term evolutionary dynamics.",
            category: "Experimental",
            params: || SimulationParams {
                paused: false,
                simulation_speed: 1,
                time_step: 1.0,
                vsync: false,
                visualization_mode: 0,
                show_extended_ui: false,
                mutation_rate: 0.5,
                predation_factor: 1.0,
                resource_diffusion: 0.08,
                resource_feed_rate: 0.012,
                resource_consumption: 0.06,
                mass_normalization_enabled: true,
                mass_damping: 0.3,
                target_mass_multiplier: 1.0,
                radius_cost_exponent: 1.3,
                agg_mobility_tradeoff: 0.3,
                starvation_severity: 0.03,
                perturbation_type: PerturbationType::None,
                perturbation_intensity: 0.5,
                perturbation_radius: 0.15,
                perturbation_active: false,
                perturbation_center_x: 0.5,
                perturbation_center_y: 0.5,
                num_seed_clusters: 30,
                seed_cluster_size: 1.0,
                initial_mass_fill: 0.15,
                seed: None,
                use_fixed_seed: false,
                fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Patch Mosaic",
            emoji: "🧩",
            description: "Patchy nutrient landscape with strong niche pressure. Favors local coexistence over global takeover.",
            category: "Experimental",
            params: || SimulationParams {
                paused: false,
                simulation_speed: 1,
                time_step: 0.9,
                vsync: false,
                visualization_mode: 0,
                show_extended_ui: false,
                mutation_rate: 0.45,
                predation_factor: 1.2,
                resource_diffusion: 0.045,
                resource_feed_rate: 0.010,
                resource_consumption: 0.075,
                mass_normalization_enabled: true,
                mass_damping: 0.42,
                target_mass_multiplier: 0.78,
                radius_cost_exponent: 1.6,
                agg_mobility_tradeoff: 0.45,
                starvation_severity: 0.05,
                perturbation_type: PerturbationType::None,
                perturbation_intensity: 0.5,
                perturbation_radius: 0.15,
                perturbation_active: false,
                perturbation_center_x: 0.5,
                perturbation_center_y: 0.5,
                num_seed_clusters: 55,
                seed_cluster_size: 0.9,
                initial_mass_fill: 0.14,
                seed: None,
                use_fixed_seed: false,
                fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Frontier Cycles",
            emoji: "🌊",
            description: "Traveling competition fronts with repeated boom-bust waves instead of static full coverage.",
            category: "Experimental",
            params: || SimulationParams {
                paused: false,
                simulation_speed: 1,
                time_step: 0.95,
                vsync: false,
                visualization_mode: 0,
                show_extended_ui: false,
                mutation_rate: 0.9,
                predation_factor: 1.8,
                resource_diffusion: 0.095,
                resource_feed_rate: 0.017,
                resource_consumption: 0.065,
                mass_normalization_enabled: true,
                mass_damping: 0.22,
                target_mass_multiplier: 1.05,
                radius_cost_exponent: 1.35,
                agg_mobility_tradeoff: 0.55,
                starvation_severity: 0.038,
                perturbation_type: PerturbationType::None,
                perturbation_intensity: 0.5,
                perturbation_radius: 0.15,
                perturbation_active: false,
                perturbation_center_x: 0.5,
                perturbation_center_y: 0.5,
                num_seed_clusters: 36,
                seed_cluster_size: 1.1,
                initial_mass_fill: 0.16,
                seed: None,
                use_fixed_seed: false,
                fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Island Succession",
            emoji: "🏞️",
            description: "Low-diffusion resource islands create local successions and refuges, slowing homogenization.",
            category: "Experimental",
            params: || SimulationParams {
                paused: false,
                simulation_speed: 1,
                time_step: 0.85,
                vsync: false,
                visualization_mode: 0,
                show_extended_ui: false,
                mutation_rate: 0.35,
                predation_factor: 0.9,
                resource_diffusion: 0.025,
                resource_feed_rate: 0.013,
                resource_consumption: 0.055,
                mass_normalization_enabled: true,
                mass_damping: 0.50,
                target_mass_multiplier: 0.72,
                radius_cost_exponent: 1.55,
                agg_mobility_tradeoff: 0.35,
                starvation_severity: 0.03,
                perturbation_type: PerturbationType::None,
                perturbation_intensity: 0.5,
                perturbation_radius: 0.15,
                perturbation_active: false,
                perturbation_center_x: 0.5,
                perturbation_center_y: 0.5,
                num_seed_clusters: 18,
                seed_cluster_size: 2.2,
                initial_mass_fill: 0.12,
                seed: None,
                use_fixed_seed: false,
                fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Sparse World",
            emoji: "🏜️",
            description: "Harsh desert conditions. Only the fittest survive in resource-poor lands.",
            category: "Experimental",
            params: || SimulationParams {
                paused: false,
                simulation_speed: 1,
                time_step: 0.8,
                vsync: false,
                visualization_mode: 0,
                show_extended_ui: false,
                mutation_rate: 0.4,
                predation_factor: 1.2,
                resource_diffusion: 0.04,
                resource_feed_rate: 0.006,
                resource_consumption: 0.08,
                mass_normalization_enabled: true,
                mass_damping: 0.5,
                target_mass_multiplier: 0.6,
                radius_cost_exponent: 1.5,
                agg_mobility_tradeoff: 0.4,
                starvation_severity: 0.06,
                perturbation_type: PerturbationType::None,
                perturbation_intensity: 0.5,
                perturbation_radius: 0.15,
                perturbation_active: false,
                perturbation_center_x: 0.5,
                perturbation_center_y: 0.5,
                num_seed_clusters: 15,
                seed_cluster_size: 1.0,
                initial_mass_fill: 0.10,
                seed: None,
                use_fixed_seed: false,
                fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Rich Paradise",
            emoji: "🏝️",
            description: "Abundant resources! Everything grows easily. Watch explosion of diversity.",
            category: "Experimental",
            params: || SimulationParams {
                paused: false,
                simulation_speed: 1,
                time_step: 1.1,
                vsync: false,
                visualization_mode: 0,
                show_extended_ui: false,
                mutation_rate: 1.0,
                predation_factor: 0.6,
                resource_diffusion: 0.12,
                resource_feed_rate: 0.028,
                resource_consumption: 0.025,
                mass_normalization_enabled: false,
                mass_damping: 0.1,
                target_mass_multiplier: 1.5,
                radius_cost_exponent: 0.85,
                agg_mobility_tradeoff: 0.1,
                starvation_severity: 0.005,
                perturbation_type: PerturbationType::None,
                perturbation_intensity: 0.5,
                perturbation_radius: 0.15,
                perturbation_active: false,
                perturbation_center_x: 0.5,
                perturbation_center_y: 0.5,
                num_seed_clusters: 25,
                seed_cluster_size: 1.5,
                initial_mass_fill: 0.12,
                seed: None,
                use_fixed_seed: false,
                fixed_seed_value: 42,
            },
        },
        PresetInfo {
            name: "Speed Demon",
            emoji: "⚡",
            description: "Maximum simulation speed! Fast time step, rapid evolution cycles.",
            category: "Experimental",
            params: || SimulationParams {
                paused: false,
                simulation_speed: 3,
                time_step: 2.0,
                vsync: false,
                visualization_mode: 0,
                show_extended_ui: false,
                mutation_rate: 1.2,
                predation_factor: 1.0,
                resource_diffusion: 0.10,
                resource_feed_rate: 0.018,
                resource_consumption: 0.05,
                mass_normalization_enabled: true,
                mass_damping: 0.2,
                target_mass_multiplier: 1.2,
                radius_cost_exponent: 1.1,
                agg_mobility_tradeoff: 0.2,
                starvation_severity: 0.025,
                perturbation_type: PerturbationType::None,
                perturbation_intensity: 0.5,
                perturbation_radius: 0.15,
                perturbation_active: false,
                perturbation_center_x: 0.5,
                perturbation_center_y: 0.5,
                num_seed_clusters: 40,
                seed_cluster_size: 1.0,
                initial_mass_fill: 0.15,
                seed: None,
                use_fixed_seed: false,
                fixed_seed_value: 42,
            },
        },
    ]
}

/// Get unique categories from the catalog
pub fn preset_categories() -> Vec<&'static str> {
    vec![
        "Autonomous",
        "Predation",
        "Explosive",
        "Stable",
        "Special",
        "Experimental",
    ]
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

pub(crate) fn load_preset(name: &str) -> Option<SimulationParams> {
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
