// ============================================================================
// app/input.rs — Keyboard state tracking and event handling
// ============================================================================

use winit::keyboard::{Key, NamedKey};

use crate::config::{TIME_STEP_MAX, TIME_STEP_MIN, VIS_MODE_COUNT};

use super::AppState;

/// Tracks which navigation keys are currently held down.
#[derive(Default)]
pub struct KeysHeld {
    pub w: bool,
    pub s: bool,
    pub a: bool,
    pub d: bool,
    pub q: bool,
    pub e: bool,
}

/// Handle keyboard events for simulation controls and global hotkeys.
pub fn handle_keyboard(
    state: &mut AppState,
    _event_loop: &winit::event_loop::ActiveEventLoop,
    event: &winit::event::KeyEvent,
    egui_consumed: bool,
) {
    let pressed = event.state.is_pressed();

    // Global hotkeys — always handled, even when egui has focus
    match &event.logical_key {
        Key::Named(NamedKey::Escape) if pressed => {
            state.lab.show_lab_ui = !state.lab.show_lab_ui;
        }
        Key::Named(NamedKey::F1) if pressed => {
            state.lab.show_lab_ui = !state.lab.show_lab_ui;
            log::info!("Lab UI: {}", if state.lab.show_lab_ui { "ON" } else { "OFF" });
        }
        Key::Named(NamedKey::F9) if pressed => {
            state.lab.show_analysis_panel = !state.lab.show_analysis_panel;
        }
        Key::Named(NamedKey::F12) if pressed => {
            state.lab.screenshot_requested = true;
            state.lab.log_event(state.world.frame, "SCREENSHOT", "Screenshot requested (F12)");
        }
        _ => {}
    }

    // Simulation controls — only if egui didn't consume the event
    if egui_consumed {
        return;
    }

    match &event.logical_key {
        Key::Named(NamedKey::Space) if pressed => {
            state.sim_params.paused = !state.sim_params.paused;
            state.lab.log_event(
                state.world.frame, "CONTROL",
                if state.sim_params.paused { "Paused" } else { "Resumed" },
            );
        }

        Key::Character(c) => match c.as_str() {
            "w" | "W" => state.keys.w = pressed,
            "s" | "S" => state.keys.s = pressed,
            "a" | "A" => state.keys.a = pressed,
            "d" | "D" => state.keys.d = pressed,
            "q" | "Q" => state.keys.q = pressed,
            "e" | "E" => state.keys.e = pressed,
            "r" | "R" if pressed => {
                state.lab.restart_requested = true;
            }
            "h" | "H" if pressed => {
                state.lab.hud_mode = if state.lab.hud_mode == 0 { 2 } else { 0 };
                let mode_name = match state.lab.hud_mode {
                    0 => "off", 2 => "NES", _ => "?",
                };
                log::info!("HUD mode: {}", mode_name);
            }
            "1" if pressed => state.sim_params.visualization_mode = 0,
            "2" if pressed => state.sim_params.visualization_mode = 1,
            "3" if pressed => state.sim_params.visualization_mode = 2,
            "4" if pressed => state.sim_params.visualization_mode = 3,
            "5" if pressed => state.sim_params.visualization_mode = 4,
            "v" | "V" if pressed => {
                state.sim_params.vsync = !state.sim_params.vsync;
                let mode = if state.sim_params.vsync {
                    wgpu::PresentMode::AutoVsync
                } else {
                    wgpu::PresentMode::Immediate
                };
                state.surface_config.present_mode = mode;
                state.surface.configure(&state.device, &state.surface_config);
            }
            "[" if pressed => {
                state.sim_params.mutation_rate = (state.sim_params.mutation_rate * 0.9).max(0.1);
            }
            "]" if pressed => {
                state.sim_params.mutation_rate = (state.sim_params.mutation_rate * 1.1).min(5.0);
            }
            _ => {}
        },

        Key::Named(named) => match named {
            NamedKey::Tab if pressed => {
                state.sim_params.visualization_mode =
                    (state.sim_params.visualization_mode + 1) % VIS_MODE_COUNT;
            }
            NamedKey::ArrowUp if pressed => {
                state.sim_params.time_step = (state.sim_params.time_step * 1.1).min(TIME_STEP_MAX);
            }
            NamedKey::ArrowDown if pressed => {
                state.sim_params.time_step = (state.sim_params.time_step * 0.9).max(TIME_STEP_MIN);
            }
            NamedKey::ArrowRight if pressed => {
                state.sim_params.simulation_speed = (state.sim_params.simulation_speed + 1).min(20);
            }
            NamedKey::ArrowLeft if pressed => {
                state.sim_params.simulation_speed =
                    state.sim_params.simulation_speed.saturating_sub(1).max(1);
            }
            _ => {}
        },

        _ => {}
    }
}
