// ============================================================================
// lab_ui/mod.rs — EvoLenia v2 Research Lab UI
// Complete egui-based interface for simulation control, parameter tuning,
// metrics visualization, experiment management, and data export.
// ============================================================================

pub mod analysis;
pub mod dashboard;
pub mod left_panel;
pub mod logs;
pub mod menu;
pub mod overlay;
pub mod presets;
pub mod status;

use crate::config::SimulationParams;
use crate::lab::LabState;

/// Main entry point for rendering all Research Lab UI panels.
pub fn render_lab_ui(ctx: &egui::Context, params: &mut SimulationParams, lab: &mut LabState) {
    if !lab.show_lab_ui {
        overlay::render_minimal_overlay(ctx, params, lab);
        return;
    }

    menu::render_menu_bar(ctx, params, lab);
    left_panel::render_left_panel(ctx, params, lab);

    if lab.show_analysis_panel {
        analysis::render_right_analysis_panel(ctx, lab);
    }

    if lab.show_logs_panel {
        logs::render_bottom_logs_panel(ctx, lab);
    }

    status::render_status_bar(ctx, lab);
}
