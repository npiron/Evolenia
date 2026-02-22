// ============================================================================
// main.rs — EvoLenia v2
// Entry point. Initializes logging and starts the event loop.
// ============================================================================

mod app;
mod camera;
mod config;
mod headless;
mod input;
mod lab;
mod lab_ui;
mod metrics;
mod pipeline;
mod renderer;
mod state_io;
mod world;

use app::{App, AppConfig};
use headless::{run_headless, HeadlessConfig};
use winit::event_loop::EventLoop;

fn main() {
    env_logger::init();

    let cli = CliOptions::from_args(std::env::args().collect());

    if cli.headless || cli.headless_then_gui {
        let headless_cfg = HeadlessConfig {
            frames: cli.frames,
            load_state_path: cli.load_state_path.clone(),
            save_state_path: Some(cli.save_state_path.clone()),
            progress_interval: cli.progress_interval,
        };
        if let Err(err) = run_headless(&headless_cfg) {
            eprintln!("Headless run failed: {err}");
            std::process::exit(1);
        }
        if !cli.headless_then_gui {
            return;
        }
    }

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = App::new(AppConfig {
        initial_state_path: if cli.headless_then_gui {
            Some(cli.save_state_path)
        } else {
            cli.load_state_path
        },
        diag_interval: cli.diag_interval,
    });
    event_loop.run_app(&mut app).unwrap();
}

#[derive(Clone, Debug)]
struct CliOptions {
    headless: bool,
    headless_then_gui: bool,
    frames: u32,
    load_state_path: Option<String>,
    save_state_path: String,
    diag_interval: u32,
    progress_interval: u32,
}

impl Default for CliOptions {
    fn default() -> Self {
        Self {
            headless: false,
            headless_then_gui: false,
            frames: 10_000,
            load_state_path: None,
            save_state_path: String::from("/tmp/evolenia_final.snap"),
            diag_interval: 300,
            progress_interval: 1000,
        }
    }
}

impl CliOptions {
    fn from_args(args: Vec<String>) -> Self {
        let mut options = Self::default();
        let mut i = 1usize;
        while i < args.len() {
            match args[i].as_str() {
                "--headless" => options.headless = true,
                "--headless-then-gui" => options.headless_then_gui = true,
                "--frames" => {
                    if i + 1 < args.len() {
                        if let Ok(v) = args[i + 1].parse::<u32>() {
                            options.frames = v.max(1);
                        }
                        i += 1;
                    }
                }
                "--load" => {
                    if i + 1 < args.len() {
                        options.load_state_path = Some(args[i + 1].clone());
                        i += 1;
                    }
                }
                "--save" => {
                    if i + 1 < args.len() {
                        options.save_state_path = args[i + 1].clone();
                        i += 1;
                    }
                }
                "--diag-interval" => {
                    if i + 1 < args.len() {
                        if let Ok(v) = args[i + 1].parse::<u32>() {
                            options.diag_interval = v.max(1);
                        }
                        i += 1;
                    }
                }
                "--progress-interval" => {
                    if i + 1 < args.len() {
                        if let Ok(v) = args[i + 1].parse::<u32>() {
                            options.progress_interval = v.max(1);
                        }
                        i += 1;
                    }
                }
                _ => {}
            }
            i += 1;
        }
        options
    }
}
