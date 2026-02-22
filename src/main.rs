// ============================================================================
// main.rs — EvoLenia v2
// Entry point. Initializes logging and starts the event loop.
// ============================================================================

mod app;
mod camera;
mod config;
mod input;
mod metrics;
mod pipeline;
mod renderer;
mod world;

use app::App;
use winit::event_loop::EventLoop;

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
