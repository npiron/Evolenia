# 🌌 EvoLenia v2.0

[![CI](https://github.com/npiron/Evolenia/actions/workflows/ci.yml/badge.svg)](https://github.com/npiron/Evolenia/actions/workflows/ci.yml)

**Spatially-Varying Continuous Cellular Automaton with Endogenous Evolution**

EvoLenia is an artificial life simulation where evolution emerges from physical laws, not from hand-coded rules. Each pixel is an autonomous organism with its own **5-gene genome**, creating a colorful ecosystem where species, predation, and adaptation emerge spontaneously.

> 💡 *Real-time evolution: Each color represents a different genetic lineage*  
> *Screenshots available in the [GitHub repository](https://github.com/npiron/Evolenia)*

---

## 🎯 What Makes This Special?

Unlike traditional cellular automata (Conway's Game of Life, Lenia):
- **No global rules** — Each organism has its own genetic code (perception radius, growth function, aggressivity, mutation rate)
- **Mass conservation** — Matter is transferred, never created or destroyed (real physics)
- **Stochastic DNA segregation** — Genes travel with mass during predation/colonization
- **Emergent speciation** — Distinct species arise without explicit speciation code
- **GPU-accelerated** — 1024×1024 organisms running at 60 FPS on modern hardware

**Technology**: Rust + WGPU (Vulkan/Metal/DX12) — Deterministic, portable, blazing fast.

---

## 🚀 Quick Start

### Prerequisites
- **Rust** (1.75+): Install from [rustup.rs](https://rustup.rs/)
- **GPU** with Vulkan/Metal/DX12 support

### Build & Run
```bash
# Clone the repository
git clone https://github.com/npiron/Evolenia.git
cd Evolenia

# Run in release mode (required for good performance)
cargo run --release
```

The simulation window will open immediately. Wait a few seconds for complex patterns to emerge.

### Fast Long Runs (Headless → GUI)
Use `run.sh` to simplify batch + replay workflows:

```bash
# Normal GUI
./run.sh gui

# Headless long run, save final state
./run.sh headless 500000 /tmp/evo_long.snap

# Headless then open the final state directly in GUI
./run.sh headless-view 200000 /tmp/evo_final.snap

# Re-open a saved state in GUI
./run.sh replay /tmp/evo_final.snap

# One-command long experiment (auto timestamped snapshot + log)
./run.sh experiment 5000000 baselineA

# Same experiment with default frames (5M)
./run.sh experiment baselineA

# Same as experiment, then opens GUI replay automatically
./run.sh experiment-view 5000000 baselineA
```

`experiment` writes outputs to `/tmp/evolenia_runs/` with automatic names:
- `baselineA_YYYYMMDD_HHMMSS.snap`
- `baselineA_YYYYMMDD_HHMMSS.log`

At the end, it prints the exact replay command to open the final state in GUI.
`experiment-view` opens that replay automatically.

Equivalent raw CLI:

```bash
cargo run --release -- --headless --frames 500000 --save /tmp/evo.snap
cargo run --release -- --load /tmp/evo.snap
```

---

## 🎮 Controls

| Key/Action         | Effect                                    |
|--------------------|-------------------------------------------|
| **WASD**           | Pan camera across the world               |
| **Q / E**          | Zoom out / Zoom in                        |
| **Mouse Wheel**    | Zoom in/out                               |
| **Space**          | Pause/Resume simulation                   |
| **R**              | Restart with new random seed              |
| **H**              | Toggle Extended HUD (shows all parameters)|
| **1-5 / Tab**      | Change visualization mode (see below)     |
| **↑ / ↓**          | Increase/Decrease time step (0.1x - 10.0x) |
| **← / →**          | Decrease/Increase simulation speed (1-10x)|
| **[ / ]**          | Decrease/Increase mutation rate (0.1x - 5.0x)|
| **ESC**            | Quit                                      |

### Extended HUD (Press H)
The extended HUD displays:
- Real-time FPS and frame counter
- Current visualization mode with quick reference
- All adjustable parameters (speed, time step, mutation rate)
- Camera position and zoom level
- World dimensions and target mass
- Quick reference for all keyboard controls

---

## 🎨 Visualization Modes

Press **1-5** to cycle through:

1. **Species Color** (default): RGB = genome(radius, μ, σ), orange glow = predators
2. **Energy Heatmap**: Blue = starving, Red = well-fed
3. **Mass Density**: Grayscale intensity
4. **Genetic Diversity**: Hue varies by local genome variance
5. **Predator/Prey**: Red = high aggressivity, Green = passive

---

## 🧬 The Science

### Five-Gene Genome
Each pixel carries:
- **`r`** — Perception radius [2-9]: How far it "sees" neighbors
- **`μ`** — Growth center [0-1]: Optimal density for survival (ecological niche)
- **`σ`** — Growth tolerance [0.01-0.3]: Generalist (high) vs specialist (low)
- **`aggressivity`** [0-1]: Predation strength (steals mass from neighbors)
- **`mutation_rate`** [0.001-0.01]: Self-modifying evolutionary instability

### Physics Engine
1. **Lenia Convolution** — Each cell convolves its neighborhood with a ring kernel to compute local density
2. **Growth Function** — Gaussian bell curve: `G(u; μ, σ) = exp(-((u - μ)² / 2σ²))`
3. **Advection** — Mass flows down/up gradients (predators chase prey)
4. **Metabolism** — Energy cost = (genome complexity + radius + aggressivity penalties) × mass
5. **Resources** — Reaction-diffusion nutrients (Gray-Scott dynamics)
6. **Mutations** — Gaussian noise applied every frame, modulated by `mutation_rate`

**Conservation Law**: Total mass remains constant (±0.01% tolerance) via normalization pass.

### Emergent Behaviors Observed
- **Speciation** — Clusters of similar genomes (species) spontaneously form
- **Predator-Prey Cycles** — High-aggressivity organisms hunt low-aggressivity ones
- **Arms Race** — Prey evolve higher `σ` (tolerance) to escape predators
- **Metastable Diversity** — System maintains 5-12 distinct species over time

See [INI.MD](INI.MD) for full mathematical formalism.

---

## 📊 Metrics & Logging

Every 300 frames, the simulation logs:
- **Frame counter**
- **Total mass** (should stay ~constant)
- **Genetic entropy** (Shannon entropy of genome distribution)
- **Number of species** (k-means clustering on genome space)

Logs are written to `stderr` in CSV format for easy plotting:
```
frame,300,target_mass,157286.4,entropy,2.456,species,7
frame,600,target_mass,157286.4,entropy,2.512,species,8
```

---

## 🏗️ Architecture

```
src/
├── main.rs                       # Entry point, CLI parsing
├── app/                          # Application: window, input, rendering, theme
│   ├── mod.rs                    # AppState, winit event loop
│   ├── gpu.rs                    # GPU device/surface init
│   ├── input.rs                  # Keyboard handling
│   ├── render.rs                 # Frame rendering + simulation encoding
│   └── theme.rs                  # egui theme (macOS-inspired light)
├── world/                        # World state & GPU buffer management
│   ├── constants.rs              # Grid size, DT, target fill
│   ├── init.rs                   # WorldState creation + init patterns
│   ├── state.rs                  # Runtime methods (snapshot, swap, uniforms, perturbations)
│   └── types.rs                  # GPU uniform structs
├── pipeline.rs                   # GPU pipeline + bind group creation
├── camera.rs                     # Camera pan/zoom
├── config.rs                     # SimulationParams, TOML config
├── renderer.rs                   # HUD text rendering (glyphon)
├── headless.rs                   # Headless batch runner
├── lab.rs                        # Research Lab state & metrics
├── lab_ui/                       # egui UI panels
│   ├── dashboard.rs, left_panel.rs, analysis.rs, menu.rs
│   ├── overlay.rs, logs.rs, presets.rs, status.rs
├── metrics.rs                    # Diagnostics (entropy, species, trophic)
├── state_io.rs                   # Binary snapshot save/load
└── shaders/
    ├── compute_velocity.wgsl      # Calculates mass flow from gradients
    ├── compute_evolution.wgsl     # Lenia + metabolism + advection + DNA + mutations
    ├── compute_resources.wgsl     # Gray-Scott reaction-diffusion for nutrients
    ├── normalize_mass.wgsl        # Conservation law enforcement (sum + normalize)
    └── render.wgsl                # 9 visualization modes
```

**Pipeline** (60 FPS):
1. Compute velocity field from mass gradients
2. Evolution pass (Lenia rule + advection + mutations)
3. Resource dynamics (nutrient diffusion)
4. Mass normalization (ensure conservation)
5. Render to screen + HUD overlay

---

## 🔬 Experimental Parameters

Want to tweak the simulation? Edit [src/world/constants.rs](src/world/constants.rs):

```rust
pub const WORLD_WIDTH: u32 = 1024;    // Grid size (power of 2)
pub const WORLD_HEIGHT: u32 = 1024;
pub const DT: f32 = 0.05;             // Time step (lower = more stable)
pub const TARGET_FILL: f32 = 0.15;    // Initial mass density (15%)
```

Or shader constants in [src/shaders/compute_evolution.wgsl](src/shaders/compute_evolution.wgsl):
- `MUTATION_STRENGTH` — Base mutation magnitude
- `PREDATION_EFFICIENCY` — Mass transfer rate during hunting
- `METABOLISM_COST` — Energy drain per frame

---

## 🐛 Troubleshooting

**Low FPS (<30)?**
- Reduce `WORLD_WIDTH/HEIGHT` to 512×512 in `src/world/constants.rs`
- Check GPU drivers are up-to-date
- Ensure you're running with `--release` flag

**Simulation dies out?**
- Increase `TARGET_FILL` to 0.25 (more initial organisms)
- Decrease mutation rate in shader (line 165, scale factor)

**All organisms look the same color?**
- Give it time (evolution takes ~1000 frames to diversify)
- Press `R` to restart with a different seed

---

## 📚 References

This project builds on:
- **Lenia** (Chan, 2019): *Lenia - Biology of Artificial Life*
- **Flow-Lenia** (2025): *Emergent Evolutionary Dynamics in Mass Conservative Lenia*, MIT Press
- **WGPU**: [wgpu.rs](https://wgpu.rs/) — Safe, portable GPU programming in Rust

---

## 📜 License

MIT License — See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions welcome! Ideas for improvement:
- [x] Real-time parameter adjustment UI
- [x] State save/load system
- [ ] Multi-GPU support for 4K simulations
- [ ] WebGPU build (run in browser)
- [ ] Phylogenetic tree visualization
- [ ] Export time-lapse videos

Open an issue or PR on [GitHub](https://github.com/npiron/Evolenia).

---

**Built with 🦀 Rust and ❤️ for artificial life research.**