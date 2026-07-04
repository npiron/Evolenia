# Presets (v1 — deprecated)

These JSON files are legacy v1 presets and may lack newer fields
introduced in v2.1.0 (e.g. `radius_cost_exponent`, `agg_mobility_tradeoff`,
`starvation_severity`, `perturbation_type`, `prey_fraction`).

**Current presets are defined in** `src/lab_ui/presets.rs` (22 built-in
presets across 6 categories). These JSON files are kept for backward
compatibility with saved snapshots from older versions.

To create a v2 preset, use the "Save Preset" button in the Research Lab UI.
