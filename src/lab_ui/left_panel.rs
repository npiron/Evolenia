// ============================================================================
// lab_ui/left_panel.rs — Left side panel with dashboard + advanced settings
// ============================================================================

use crate::config::{PerturbationType, SimulationParams, TIME_STEP_MAX, TIME_STEP_MIN};
use crate::lab::LabState;
use crate::world::target_total_mass;

use super::dashboard::render_dashboard;
use super::presets::{self, preset_catalog, preset_categories, PresetInfo};

pub fn render_left_panel(ctx: &egui::Context, params: &mut SimulationParams, lab: &mut LabState) {
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
                        .color(egui::Color32::from_rgb(0, 100, 200)),
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
                render_params_section(ui, params, lab);
                ui.separator();
                render_perturbation_section(ui, params, lab);
                ui.separator();
                render_experiment_section(ui, params, lab);
                ui.add_space(10.0);
            });
        });
}

// ======================== Control Section ========================

fn render_control_section(ui: &mut egui::Ui, params: &mut SimulationParams, lab: &mut LabState) {
    egui::CollapsingHeader::new(
        egui::RichText::new("⏱ Time & Speed")
            .strong()
            .color(egui::Color32::from_rgb(20, 30, 50)),
    )
    .default_open(false)
    .show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label("Time Step:");
            if slider_with_range(
                ui,
                egui::Slider::new(&mut params.time_step, TIME_STEP_MIN..=TIME_STEP_MAX)
                    .step_by(0.05)
                    .text("dt"),
                TIME_STEP_MIN,
                TIME_STEP_MAX,
            )
            .on_hover_text("Simulation time step — smaller = more stable but slower evolution")
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
            .color(egui::Color32::from_rgb(30, 120, 50)),
        );
    });
}

// ======================== Parameters Section ========================

fn section_card<R>(
    ui: &mut egui::Ui,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) -> egui::InnerResponse<R> {
    egui::Frame::new()
        .fill(egui::Color32::TRANSPARENT)
        .stroke(egui::Stroke::new(
            1.0,
            egui::Color32::from_rgb(200, 205, 215),
        ))
        .corner_radius(4)
        .inner_margin(egui::Margin::symmetric(8, 6))
        .outer_margin(egui::Margin::symmetric(0, 2))
        .show(ui, add_contents)
}

fn render_params_section(ui: &mut egui::Ui, params: &mut SimulationParams, lab: &mut LabState) {
    egui::CollapsingHeader::new(
        egui::RichText::new("🧬 Paramètres avancés")
            .strong()
            .color(egui::Color32::from_rgb(20, 30, 50)),
    )
    .default_open(false)
    .show(ui, |ui| {
        section_card(ui, |ui| {
            ui.label(
                egui::RichText::new("Évolution / Mutation")
                    .strong()
                    .color(egui::Color32::from_rgb(20, 30, 50)),
            );
            if slider_with_range(
                ui,
                egui::Slider::new(&mut params.mutation_rate, 0.1..=5.0)
                    .text("Taux de mutation")
                    .step_by(0.1),
                0.1, 5.0,
            ).on_hover_text("Force du bruit génétique à chaque frame. Élevé = plus de diversification")
                .changed() {
                lab.log_event(0, "PARAM_CHANGE", &format!("mutation_rate={:.1}", params.mutation_rate));
            }
        });

        section_card(ui, |ui| {
            ui.label(
                egui::RichText::new("Prédation")
                    .strong()
                    .color(egui::Color32::from_rgb(20, 30, 50)),
            );
            if slider_with_range(
                ui,
                egui::Slider::new(&mut params.predation_factor, 0.0..=3.0)
                    .text("Facteur de prédation")
                    .step_by(0.1),
                0.0, 3.0,
            ).on_hover_text("Efficacité du transfert de masse lors d'une attaque. 0 = pas de prédation")
                .changed() {
                lab.log_event(0, "PARAM_CHANGE", &format!("predation={:.1}", params.predation_factor));
            }
        });

        section_card(ui, |ui| {
            ui.label(
                egui::RichText::new("Ressources (Gray-Scott)")
                    .strong()
                    .color(egui::Color32::from_rgb(20, 30, 50)),
            );
            if slider_with_range(
                ui,
                egui::Slider::new(&mut params.resource_diffusion, 0.0..=0.5)
                    .text("Diffusion")
                    .step_by(0.01),
                0.0, 0.5,
            ).on_hover_text("Vitesse de propagation des nutriments dans l'espace")
                .changed() {
                lab.log_event(0, "PARAM_CHANGE", &format!("diffusion={:.3}", params.resource_diffusion));
            }
            if slider_with_range(
                ui,
                egui::Slider::new(&mut params.resource_feed_rate, 0.0..=0.1)
                    .text("Apport")
                    .step_by(0.001),
                0.0, 0.1,
            ).on_hover_text("Taux d'arrivée de nouveaux nutriments")
                .changed() {
                lab.log_event(0, "PARAM_CHANGE", &format!("feed_rate={:.4}", params.resource_feed_rate));
            }
            if slider_with_range(
                ui,
                egui::Slider::new(&mut params.resource_consumption, 0.0..=0.3)
                    .text("Consommation")
                    .step_by(0.01),
                0.0, 0.3,
            ).on_hover_text("Quantité de nutriments consommés par les organismes")
                .changed() {
                lab.log_event(0, "PARAM_CHANGE", &format!("consumption={:.3}", params.resource_consumption));
            }
        });

        section_card(ui, |ui| {
            ui.label(
                egui::RichText::new("Normalisation de masse")
                    .strong()
                    .color(egui::Color32::from_rgb(20, 30, 50)),
            );
            if ui.checkbox(&mut params.mass_normalization_enabled, "Activée")
                .on_hover_text("Maintient la masse totale constante dans le monde.\nDésactiver = système ouvert : la masse peut dériver.")
                .changed() {
                lab.log_event(0, "PARAM_CHANGE", &format!("norm_enabled={}", params.mass_normalization_enabled));
            }
            if !params.mass_normalization_enabled {
                ui.label(
                    egui::RichText::new("⚠️ Système ouvert : la masse totale n'est plus conservée.")
                        .small()
                        .color(egui::Color32::from_rgb(180, 130, 20)),
                );
            }
            if params.mass_normalization_enabled {
                if slider_with_range(
                    ui,
                    egui::Slider::new(&mut params.mass_damping, 0.05..=1.0)
                        .text("Amortissement")
                        .step_by(0.05),
                    0.05, 1.0,
                ).on_hover_text("Force du rappel vers la masse cible. 1.0 = correction totale immédiate")
                    .changed() {
                    lab.log_event(0, "PARAM_CHANGE", &format!("damping={:.2}", params.mass_damping));
                }
                if slider_with_range(
                    ui,
                    egui::Slider::new(&mut params.target_mass_multiplier, 0.1..=3.0)
                        .text("Masse cible ×")
                        .step_by(0.1),
                    0.1, 3.0,
                ).on_hover_text("Multiplicateur de la masse totale cible")
                    .changed() {
                    lab.log_event(0, "PARAM_CHANGE", &format!("target_mass_mult={:.1}", params.target_mass_multiplier));
                }
                ui.label(
                    egui::RichText::new(format!("Cible: {:.0}", target_total_mass() * params.target_mass_multiplier))
                        .small()
                        .color(egui::Color32::from_rgb(30, 120, 50)),
                );
            }
        });

        section_card(ui, |ui| {
            ui.label(
                egui::RichText::new("⚖ Trade-offs")
                    .strong()
                    .color(egui::Color32::from_rgb(20, 30, 50)),
            );
            if slider_with_range(
                ui,
                egui::Slider::new(&mut params.radius_cost_exponent, 1.0..=3.0)
                    .text("Coût du radius")
                    .step_by(0.1),
                1.0, 3.0,
            ).on_hover_text("Pénalité exponentielle pour les grands rayons de perception")
                .changed() {
                lab.log_event(0, "PARAM_CHANGE", &format!("radius_cost_exp={:.1}", params.radius_cost_exponent));
            }
            if slider_with_range(
                ui,
                egui::Slider::new(&mut params.agg_mobility_tradeoff, 0.0..=1.0)
                    .text("Aggressivité ↔ Mobilité")
                    .step_by(0.05),
                0.0, 1.0,
            ).on_hover_text("Plus un organisme est agressif, moins il se déplace vite")
                .changed() {
                lab.log_event(0, "PARAM_CHANGE", &format!("agg_mobility={:.2}", params.agg_mobility_tradeoff));
            }
            if slider_with_range(
                ui,
                egui::Slider::new(&mut params.starvation_severity, 0.01..=0.2)
                    .text("Sévérité famine")
                    .step_by(0.005),
                0.01, 0.2,
            ).on_hover_text("Pénalité énergétique quand les ressources sont basses")
                .changed() {
                lab.log_event(0, "PARAM_CHANGE", &format!("starvation={:.3}", params.starvation_severity));
            }
        });

        section_card(ui, |ui| {
            ui.label(
                egui::RichText::new("Conditions initiales (au restart)")
                    .strong()
                    .color(egui::Color32::from_rgb(20, 30, 50)),
            );
            islider_with_range(
                ui,
                egui::Slider::new(&mut params.num_seed_clusters, 5..=100)
                    .text("Clusters de départ"),
                5, 100,
            ).on_hover_text("Nombre de groupes d'organismes initiaux");
            slider_with_range(
                ui,
                egui::Slider::new(&mut params.seed_cluster_size, 0.5..=3.0)
                    .text("Taille des clusters")
                    .step_by(0.1),
                0.5, 3.0,
            );
            slider_with_range(
                ui,
                egui::Slider::new(&mut params.initial_mass_fill, 0.05..=0.5)
                    .text("Remplissage initial %")
                    .step_by(0.01),
                0.05, 0.5,
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
    egui::CollapsingHeader::new(
        egui::RichText::new("🌊 Perturbations")
            .strong()
            .color(egui::Color32::from_rgb(20, 30, 50)),
    )
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

        slider_with_range(
            ui,
            egui::Slider::new(&mut params.perturbation_intensity, 0.0..=1.0)
                .text("Intensité")
                .step_by(0.05),
            0.0,
            1.0,
        )
        .on_hover_text("Force de la perturbation");

        slider_with_range(
            ui,
            egui::Slider::new(&mut params.perturbation_radius, 0.05..=0.5)
                .text("Rayon")
                .step_by(0.01),
            0.05,
            0.5,
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
                .color(egui::Color32::from_rgb(140, 130, 30)),
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
                egui::RichText::new("● En attente…").color(egui::Color32::from_rgb(200, 150, 20)),
            );
        }
    });
}

// ======================== Experiment Section ========================

fn render_experiment_section(ui: &mut egui::Ui, params: &mut SimulationParams, lab: &mut LabState) {
    egui::CollapsingHeader::new(
        egui::RichText::new("🧪 Expériences")
            .strong()
            .color(egui::Color32::from_rgb(20, 30, 50)),
    )
    .default_open(false)
    .show(ui, |ui| {
        // Seed control
        section_card(ui, |ui| {
            ui.label(
                egui::RichText::new("Reproducibility")
                    .strong()
                    .color(egui::Color32::from_rgb(20, 30, 50)),
            );
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
                        .color(egui::Color32::from_rgb(30, 120, 50)),
                );
            }
        });

        // Run management
        section_card(ui, |ui| {
            ui.label(
                egui::RichText::new("Run Management")
                    .strong()
                    .color(egui::Color32::from_rgb(20, 30, 50)),
            );
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
                    egui::RichText::new("● Recording").color(egui::Color32::from_rgb(30, 180, 40)),
                );
            }

            ui.label(format!("Metrics: {} samples", lab.metrics_history.len()));
        });

        // Presets - Full Selector System
        section_card(ui, |ui| {
            ui.label(
                egui::RichText::new("🎛️ Preset Selector")
                    .strong()
                    .size(14.0)
                    .color(egui::Color32::from_rgb(20, 30, 50)),
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

                    let first_in_cat = catalog.iter().position(|p| p.category == *cat).unwrap_or(0);

                    let is_selected = catalog
                        .get(lab.selected_preset_index)
                        .map(|p| p.category == *cat)
                        .unwrap_or(cat_idx == 0);

                    let btn =
                        egui::Button::new(format!("{} {}", cat_emoji, cat)).fill(if is_selected {
                            egui::Color32::from_rgb(0, 122, 255)
                        } else {
                            egui::Color32::from_rgb(225, 228, 235)
                        });

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
                                egui::Color32::from_rgb(0, 122, 255)
                            } else {
                                egui::Color32::from_rgb(232, 235, 240)
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
                                        .color(egui::Color32::from_rgb(100, 100, 115)),
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
                        let vis = params.visualization_mode;
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
                        presets::save_preset(&lab.preset_name, params);
                        lab.set_status(format!("Preset '{}' saved to disk", lab.preset_name));
                    }
                    if ui.button("Load").clicked() {
                        if let Some(loaded) = presets::load_preset(&lab.preset_name) {
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

// ======================== Slider Helpers ========================

fn slider_with_range(
    ui: &mut egui::Ui,
    slider: egui::Slider<'_>,
    range_min: f32,
    range_max: f32,
) -> egui::Response {
    let response = ui.add(slider);
    ui.label(
        egui::RichText::new(format!("[{:.3} … {:.3}]", range_min, range_max))
            .size(10.0)
            .color(egui::Color32::from_rgb(140, 140, 155)),
    );
    response
}

fn islider_with_range(
    ui: &mut egui::Ui,
    slider: egui::Slider<'_>,
    range_min: u32,
    range_max: u32,
) -> egui::Response {
    let response = ui.add(slider);
    ui.label(
        egui::RichText::new(format!("[{} … {}]", range_min, range_max))
            .size(10.0)
            .color(egui::Color32::from_rgb(140, 140, 155)),
    );
    response
}
