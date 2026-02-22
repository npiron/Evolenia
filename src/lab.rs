// ============================================================================
// lab.rs — EvoLenia v2 Research Lab
// Central state for experiment management, metrics recording, run tracking,
// screenshot capture, and data export.
// ============================================================================

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use chrono::Local;
use serde::Serialize;

use crate::config::SimulationParams;
use crate::metrics::SimDiagnostics;
use crate::world::{WORLD_HEIGHT, WORLD_WIDTH};

// ======================== Metrics Record ========================

#[derive(Clone, Debug, Serialize)]
pub struct MetricsRecord {
    pub frame: u32,
    pub time_ms: f64,
    pub fps: f32,
    pub total_mass: f32,
    pub avg_energy: f32,
    pub entropy: f32,
    pub species: usize,
    pub live_pixels: u32,
    pub live_fraction: f32,
    pub predator_fraction: f32,
    pub avg_resource: f32,
    pub mass_std_dev: f32,
    pub avg_radius: f32,
    pub avg_mu: f32,
    pub avg_sigma: f32,
    pub avg_aggressivity: f32,
    pub avg_mutation_rate: f32,
    // Phase 1 eco metrics
    pub prey_fraction: f32,
    pub opportunist_fraction: f32,
    pub effective_diversity: f32,
    pub genome_variance: f32,
    pub total_energy: f32,
    pub energy_flux: f32,
}

impl MetricsRecord {
    pub fn csv_header() -> &'static str {
        "frame,time_ms,fps,total_mass,avg_energy,entropy,species,live_pixels,live_fraction,predator_fraction,avg_resource,mass_std_dev,avg_radius,avg_mu,avg_sigma,avg_aggressivity,avg_mutation_rate,prey_fraction,opportunist_fraction,effective_diversity,genome_variance,total_energy,energy_flux"
    }

    pub fn to_csv_line(&self) -> String {
        format!(
            "{},{:.1},{:.1},{:.2},{:.4},{:.3},{},{},{:.4},{:.4},{:.4},{:.5},{:.3},{:.4},{:.4},{:.4},{:.6},{:.4},{:.4},{:.3},{:.5},{:.2},{:.5}",
            self.frame, self.time_ms, self.fps, self.total_mass, self.avg_energy,
            self.entropy, self.species, self.live_pixels, self.live_fraction,
            self.predator_fraction, self.avg_resource, self.mass_std_dev,
            self.avg_radius, self.avg_mu, self.avg_sigma,
            self.avg_aggressivity, self.avg_mutation_rate,
            self.prey_fraction, self.opportunist_fraction,
            self.effective_diversity, self.genome_variance,
            self.total_energy, self.energy_flux,
        )
    }
}

// ======================== Lab Event ========================

#[derive(Clone, Debug, Serialize)]
pub struct LabEvent {
    pub frame: u32,
    pub time_ms: f64,
    pub event_type: String,
    pub details: String,
}

impl LabEvent {
    pub fn to_log_line(&self) -> String {
        format!(
            "[{:.1}ms] frame={} {} — {}",
            self.time_ms, self.frame, self.event_type, self.details,
        )
    }
}

// ======================== Run Summary ========================

#[derive(Clone, Debug, Serialize)]
pub struct RunSummary {
    pub run_id: String,
    pub run_dir: PathBuf,
    pub start_time: String,
    pub total_frames: u32,
    pub metrics_count: usize,
}

// ======================== Lab State ========================

pub struct LabState {
    // -- Run management --
    pub run_id: String,
    pub run_start: Instant,
    pub run_start_time: String,
    pub run_dir: PathBuf,
    pub run_active: bool,

    // -- Metrics --
    pub metrics_history: Vec<MetricsRecord>,
    pub metrics_sample_interval: u32,

    // -- Events --
    pub events: Vec<LabEvent>,

    // -- UI state --
    pub show_lab_ui: bool,
    pub show_analysis_panel: bool,
    pub show_logs_panel: bool,

    // -- Actions --
    pub restart_requested: bool,
    pub step_requested: bool,
    pub screenshot_requested: bool,
    pub snapshot_requested: bool,

    // -- Comparison --
    pub completed_runs: Vec<RunSummary>,
    pub comparison_a: Option<usize>,
    pub comparison_b: Option<usize>,

    // -- Config presets --
    pub preset_name: String,

    // -- Status messages --
    pub status_message: Option<(String, Instant)>,
}

impl Default for LabState {
    fn default() -> Self {
        let now = Local::now();
        let run_id = format!("run_{}", now.format("%Y%m%d_%H%M%S"));
        let run_dir = PathBuf::from(format!(
            "runs/{}/{}",
            now.format("%Y-%m-%d"),
            &run_id
        ));

        Self {
            run_id,
            run_start: Instant::now(),
            run_start_time: now.format("%Y-%m-%d %H:%M:%S").to_string(),
            run_dir,
            run_active: false,

            metrics_history: Vec::with_capacity(10_000),
            metrics_sample_interval: 300,

            events: Vec::with_capacity(1_000),

            show_lab_ui: true,
            show_analysis_panel: false,
            show_logs_panel: true,

            restart_requested: false,
            step_requested: false,
            screenshot_requested: false,
            snapshot_requested: false,

            completed_runs: Vec::new(),
            comparison_a: None,
            comparison_b: None,

            preset_name: String::from("default"),

            status_message: None,
        }
    }
}

impl LabState {
    /// Start a new run: create output directory, save initial config.
    pub fn start_run(&mut self, params: &SimulationParams) {
        let now = Local::now();
        self.run_id = format!("run_{}", now.format("%Y%m%d_%H%M%S"));
        self.run_dir = PathBuf::from(format!(
            "runs/{}/{}",
            now.format("%Y-%m-%d"),
            &self.run_id
        ));
        self.run_start = Instant::now();
        self.run_start_time = now.format("%Y-%m-%d %H:%M:%S").to_string();
        self.run_active = true;
        self.metrics_history.clear();
        self.events.clear();

        // Create directories
        if let Err(e) = fs::create_dir_all(&self.run_dir) {
            log::error!("Failed to create run directory {:?}: {}", self.run_dir, e);
            return;
        }
        let screenshots_dir = self.run_dir.join("screenshots");
        if let Err(e) = fs::create_dir_all(&screenshots_dir) {
            log::error!("Failed to create screenshots dir: {}", e);
        }

        // Save config
        self.save_config(params);
        self.log_event(0, "RUN_START", &format!("Run {} started", self.run_id));
        self.set_status(format!("Run {} started", self.run_id));
    }

    /// Save config.json for the current run.
    pub fn save_config(&self, params: &SimulationParams) {
        let config = serde_json::json!({
            "run_id": self.run_id,
            "timestamp": self.run_start_time,
            "app_version": env!("CARGO_PKG_VERSION"),
            "world_width": WORLD_WIDTH,
            "world_height": WORLD_HEIGHT,
            "params": params,
        });

        let path = self.run_dir.join("config.json");
        match serde_json::to_string_pretty(&config) {
            Ok(json) => {
                if let Err(e) = fs::write(&path, json) {
                    log::error!("Failed to write config.json: {}", e);
                } else {
                    log::info!("Saved config to {:?}", path);
                }
            }
            Err(e) => log::error!("Failed to serialize config: {}", e),
        }
    }

    /// Record a metrics sample from GPU readback diagnostics.
    pub fn record_metrics(&mut self, diag: &SimDiagnostics, frame: u32, fps: f32) {
        let time_ms = self.run_start.elapsed().as_secs_f64() * 1000.0;
        let record = MetricsRecord {
            frame,
            time_ms,
            fps,
            total_mass: diag.total_mass,
            avg_energy: diag.avg_energy,
            entropy: diag.genetic_entropy,
            species: diag.species_count,
            live_pixels: diag.live_pixels,
            live_fraction: diag.live_fraction,
            predator_fraction: diag.genome_stats.predator_fraction,
            avg_resource: diag.avg_resource,
            mass_std_dev: diag.mass_std_dev,
            avg_radius: diag.genome_stats.avg_radius,
            avg_mu: diag.genome_stats.avg_mu,
            avg_sigma: diag.genome_stats.avg_sigma,
            avg_aggressivity: diag.genome_stats.avg_aggressivity,
            avg_mutation_rate: diag.genome_stats.avg_mutation_rate,
            prey_fraction: diag.prey_fraction,
            opportunist_fraction: diag.opportunist_fraction,
            effective_diversity: diag.effective_diversity,
            genome_variance: diag.genome_variance,
            total_energy: diag.total_energy,
            energy_flux: diag.energy_flux,
        };
        self.metrics_history.push(record);
    }

    /// Log an event.
    pub fn log_event(&mut self, frame: u32, event_type: &str, details: &str) {
        let time_ms = self.run_start.elapsed().as_secs_f64() * 1000.0;
        self.events.push(LabEvent {
            frame,
            time_ms,
            event_type: event_type.to_string(),
            details: details.to_string(),
        });
    }

    /// Export metrics to CSV.
    pub fn export_metrics_csv(&self) -> Result<PathBuf, String> {
        let path = self.run_dir.join("metrics.csv");
        let mut file = fs::File::create(&path)
            .map_err(|e| format!("Failed to create metrics.csv: {}", e))?;

        writeln!(file, "{}", MetricsRecord::csv_header())
            .map_err(|e| format!("Write error: {}", e))?;

        for record in &self.metrics_history {
            writeln!(file, "{}", record.to_csv_line())
                .map_err(|e| format!("Write error: {}", e))?;
        }

        log::info!("Exported {} metrics records to {:?}", self.metrics_history.len(), path);
        Ok(path)
    }

    /// Export events log.
    pub fn export_events_log(&self) -> Result<PathBuf, String> {
        let path = self.run_dir.join("events.log");
        let mut file = fs::File::create(&path)
            .map_err(|e| format!("Failed to create events.log: {}", e))?;

        for event in &self.events {
            writeln!(file, "{}", event.to_log_line())
                .map_err(|e| format!("Write error: {}", e))?;
        }

        log::info!("Exported {} events to {:?}", self.events.len(), path);
        Ok(path)
    }

    /// Export a full run report (markdown).
    pub fn export_report(&self, params: &SimulationParams) -> Result<PathBuf, String> {
        let path = self.run_dir.join("report.md");
        let mut file = fs::File::create(&path)
            .map_err(|e| format!("Failed to create report.md: {}", e))?;

        let last_metrics = self.metrics_history.last();

        let report = format!(
            "# EvoLenia Experiment Report\n\n\
             ## Run Info\n\
             - **Run ID**: {}\n\
             - **Start**: {}\n\
             - **Frames**: {}\n\
             - **Metrics Samples**: {}\n\
             - **App Version**: {}\n\
             - **World Size**: {}×{}\n\n\
             ## Parameters\n\
             ```json\n{}\n```\n\n\
             ## Final Metrics\n\
             {}\n\n\
             ## Events Summary\n\
             - Total events: {}\n\
             {}\n",
            self.run_id,
            self.run_start_time,
            last_metrics.map_or(0, |m| m.frame),
            self.metrics_history.len(),
            env!("CARGO_PKG_VERSION"),
            WORLD_WIDTH,
            WORLD_HEIGHT,
            serde_json::to_string_pretty(params).unwrap_or_default(),
            if let Some(m) = last_metrics {
                format!(
                    "| Metric | Value |\n|--------|-------|\n\
                     | Total Mass | {:.1} |\n\
                     | Avg Energy | {:.4} |\n\
                     | Entropy | {:.3} bits |\n\
                     | Species | {} |\n\
                     | Live Pixels | {} ({:.1}%) |\n\
                     | Predator % | {:.1}% |\n\
                     | FPS | {:.0} |",
                    m.total_mass, m.avg_energy, m.entropy, m.species,
                    m.live_pixels, m.live_fraction * 100.0,
                    m.predator_fraction * 100.0, m.fps,
                )
            } else {
                "No metrics collected.".to_string()
            },
            self.events.len(),
            self.events.iter().rev().take(10)
                .map(|e| format!("- {}", e.to_log_line()))
                .collect::<Vec<_>>().join("\n"),
        );

        write!(file, "{}", report).map_err(|e| format!("Write error: {}", e))?;
        log::info!("Exported report to {:?}", path);
        Ok(path)
    }

    /// Finalize the current run: export all data and archive.
    pub fn finalize_run(&mut self, params: &SimulationParams) {
        if !self.run_active {
            return;
        }

        let total_frames = self.metrics_history.last().map_or(0, |m| m.frame);

        // Export data
        if let Err(e) = self.export_metrics_csv() {
            log::error!("Failed to export metrics: {}", e);
        }
        if let Err(e) = self.export_events_log() {
            log::error!("Failed to export events: {}", e);
        }
        if let Err(e) = self.export_report(params) {
            log::error!("Failed to export report: {}", e);
        }

        // Save run summary for comparison
        self.completed_runs.push(RunSummary {
            run_id: self.run_id.clone(),
            run_dir: self.run_dir.clone(),
            start_time: self.run_start_time.clone(),
            total_frames,
            metrics_count: self.metrics_history.len(),
        });

        self.log_event(total_frames, "RUN_END", &format!("Run {} finalized", self.run_id));
        self.set_status(format!("Run {} finalized — data exported", self.run_id));
        self.run_active = false;
    }

    /// Save a screenshot to the run's screenshots directory.
    pub fn save_screenshot(
        &self,
        frame: u32,
        width: u32,
        height: u32,
        rgba_data: &[u8],
        vis_mode: u32,
    ) -> Result<PathBuf, String> {
        let screenshots_dir = self.run_dir.join("screenshots");
        fs::create_dir_all(&screenshots_dir)
            .map_err(|e| format!("Failed to create screenshots dir: {}", e))?;

        let filename = format!(
            "frame{:06}_{}_{}.png",
            frame,
            crate::config::visualization_mode_name(vis_mode).replace('/', "_"),
            &self.run_id,
        );
        let path = screenshots_dir.join(&filename);

        image::save_buffer(
            &path,
            rgba_data,
            width,
            height,
            image::ColorType::Rgba8,
        )
        .map_err(|e| format!("Failed to save screenshot: {}", e))?;

        log::info!("Screenshot saved: {:?}", path);
        Ok(path)
    }

    /// Set a temporary status message.
    pub fn set_status(&mut self, msg: String) {
        self.status_message = Some((msg, Instant::now()));
    }

    /// Get the current status message (auto-clears after 5 seconds).
    pub fn current_status(&mut self) -> Option<&str> {
        let should_clear = matches!(
            &self.status_message,
            Some((_, when)) if when.elapsed().as_secs() >= 5
        );
        if should_clear {
            self.status_message = None;
        }
        self.status_message.as_ref().map(|(msg, _)| msg.as_str())
    }

    /// Load metrics from a previous run CSV for comparison.
    pub fn load_comparison_metrics(path: &PathBuf) -> Result<Vec<MetricsRecord>, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {:?}: {}", path, e))?;
        let mut records = Vec::new();
        for (i, line) in content.lines().enumerate() {
            if i == 0 { continue; } // skip header
            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() < 17 { continue; }
            let record = MetricsRecord {
                frame: fields[0].parse().unwrap_or(0),
                time_ms: fields[1].parse().unwrap_or(0.0),
                fps: fields[2].parse().unwrap_or(0.0),
                total_mass: fields[3].parse().unwrap_or(0.0),
                avg_energy: fields[4].parse().unwrap_or(0.0),
                entropy: fields[5].parse().unwrap_or(0.0),
                species: fields[6].parse().unwrap_or(0),
                live_pixels: fields[7].parse().unwrap_or(0),
                live_fraction: fields[8].parse().unwrap_or(0.0),
                predator_fraction: fields[9].parse().unwrap_or(0.0),
                avg_resource: fields[10].parse().unwrap_or(0.0),
                mass_std_dev: fields[11].parse().unwrap_or(0.0),
                avg_radius: fields[12].parse().unwrap_or(0.0),
                avg_mu: fields[13].parse().unwrap_or(0.0),
                avg_sigma: fields[14].parse().unwrap_or(0.0),
                avg_aggressivity: fields[15].parse().unwrap_or(0.0),
                avg_mutation_rate: fields[16].parse().unwrap_or(0.0),
                // Phase 1 eco metrics (default 0 for backward compat with old CSVs)
                prey_fraction: fields.get(17).and_then(|s| s.parse().ok()).unwrap_or(0.0),
                opportunist_fraction: fields.get(18).and_then(|s| s.parse().ok()).unwrap_or(0.0),
                effective_diversity: fields.get(19).and_then(|s| s.parse().ok()).unwrap_or(0.0),
                genome_variance: fields.get(20).and_then(|s| s.parse().ok()).unwrap_or(0.0),
                total_energy: fields.get(21).and_then(|s| s.parse().ok()).unwrap_or(0.0),
                energy_flux: fields.get(22).and_then(|s| s.parse().ok()).unwrap_or(0.0),
            };
            records.push(record);
        }
        Ok(records)
    }
}
