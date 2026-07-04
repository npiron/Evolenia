# Changelog

All notable changes to EvoLenia will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2026-07-02

### Added
- **CI/CD pipeline** (`.github/workflows/ci.yml`): format check, clippy (`-D warnings`), tests, release build on every push/PR. Badge in README.
- **TOML config file support**: optional `--config` CLI flag to load `SimulationParams` from a `.toml` file. All fields optional, defaults applied for missing ones.
- **`justfile`** with commands: `run`, `test`, `bench`, `headless`, `fmt`, `lint`, `ci`, and more.
- **Headless seed support**: `--seed <u64>` flag for deterministic headless runs. `HeadlessConfig` now accepts an optional seed.
- **Debug raw visualization mode** (mode 8): direct RGB = mass/energy/resources, for shader debugging.
- **New integration tests**: empty grid, saturated grid, extreme parameters, half-live grid, save/load checksum roundtrip, preset smoke tests (9 presets validated). 2 `#[ignore]` GPU integration tests included.
- **`HudPrepareConfig` struct** to reduce argument count in `HudRenderer::prepare`.

### Changed
- **Structured logging**: `env_logger` now uses `RUST_LOG` env var with timestamp format. Default filter: `info`.
- **Normalize shader**: `dust_floor` is now a GPU uniform (`NormalizeParams.dust_floor_x1000`) instead of hardcoded `0.002`.
- **Render shader**: extracted `blend_over_background()` helper to deduplicate `mix(bg, X, m)` across all 8 visualization modes.
- **Render `unwrap()` → graceful fallback**: `HudRenderer::prepare` returns `bool` and logs warnings on failure instead of panicking.
- **Render `unwrap()` → logged warning**: `HudRenderer::render` logs warning instead of panicking on glyph render failure.
- **Removed `WorldState::new`**: all callers now use `new_with_config(device, seed, params)` for explicit seed control.
- **Clippy compliance**: all warnings fixed (`manual_div_ceil`, `same_item_push`, `manual_range_contains`, `assertions_on_constants`, `too_many_arguments`, `unnecessary_cast`, `is_multiple_of`).

### Fixed
- **NaN/Inf guards in all 5 WGSL shaders**:
  - `compute_evolution.wgsl`: output validation on mass/energy/genome before write.
  - `compute_velocity.wgsl`: velocity reset to zero if non-finite components.
  - `compute_resources.wgsl`: resource reset to 0.5 if NaN/Inf.
  - `normalize_mass.wgsl`: mass input validated for NaN/Inf before atomic add.
- **Normalize race condition**: documented that `actual_total` is computed identically by all threads (same `atomicLoad` value) — no race.

### Removed
- **`UI_INTEGRATION.md`**: obsolete integration guide (egui is fully integrated since v2.0).

### Technical
- 0 clippy warnings with `-D warnings` (clean lint pass).
- 55 tests total: 53 pass, 2 `#[ignore]` (GPU-dependent).
- 3 new dependencies: `toml`, `serde_spanned`, `winnow`.

---

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
