# Changelog

All notable changes to EvoLenia will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-02-22

### Added
- **Complete README** with full documentation, installation instructions, and scientific background
- **Multiple visualization modes** (Species Color, Energy Heatmap, Mass Density, Genetic Diversity, Predator/Prey)
- **Pause/Resume functionality** (Space key)
- **Restart simulation** (R key) with new random seed
- **Interactive HUD** showing frame count, FPS, visualization mode, and controls
- **Metrics module** for emergence analysis (genetic entropy, species detection, genome statistics)
- **MIT License** for open-source distribution
- **Build script** (`run.sh`) for easy compilation and execution

### Changed
- Improved shader render pipeline to support multiple visualization modes
- Enhanced RenderParams uniform to include visualization mode selection
- Updated camera controls to show current state in HUD
- Refactored energy buffer binding in render pipeline

### Fixed
- Conservation of mass now properly enforced via normalization pass
- Stochastic DNA segregation implemented correctly (no genome averaging)
- Mutation rates bounded to prevent drift to extremes
- Shader guards against division by zero (sigma > 0 check)

### Technical Details
- **5-gene genome**: radius, μ (growth center), σ (growth tolerance), aggressivity, mutation_rate
- **Gray-Scott resource dynamics**: Nutrient diffusion, regeneration, and consumption
- **Mass-conservative advection**: Total mass preserved within ±0.01% tolerance
- **GPU-accelerated**: All compute and render passes run on GPU via WGPU
- **60 FPS target** on modern hardware (1024×1024 grid)

### Known Limitations
- State save/load not yet implemented (planned for v2.1)
- Metrics logging to CSV requires manual readback (automatic logging planned for v2.1)
- WebGPU build for browser support (planned for v3.0)

---

## [1.0.0] - 2025-XX-XX (Initial Development)

### Added
- Initial Lenia implementation
- Basic GPU kernels
- Genome system prototype

### Issues (Fixed in v2.0)
- Mass was not properly conserved
- Genome averaging instead of segregation
- Missing scientific documentation
- No visualization modes
