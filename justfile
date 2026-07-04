# EvoLenia v2 — Justfile
# Development and CI commands for the EvoLenia artificial life simulation.

default: run

# ── Build & Run ──

# Run EvoLenia in GUI mode (release)
run:
    cargo run --release

# Run EvoLenia in GUI mode (debug, faster compile)
run-debug:
    cargo run

# ── Testing ──

# Run all unit tests
test:
    cargo test --release

# Run tests with output (show println! and log)
test-verbose:
    cargo test --release -- --nocapture

# Run only a specific test module
test-module module:
    cargo test --release {{module}}

# ── Headless ──

# Run 1000 frames headless (quick smoke test)
headless frames="1000":
    cargo run --release -- --headless --frames {{frames}} --progress-interval 500

# Run headless with specific seed for reproducibility
headless-seed frames="5000" seed="42":
    cargo run --release -- --headless --frames {{frames}} --seed {{seed}} --progress-interval 1000

# Run headless then open GUI
headless-gui frames="5000":
    cargo run --release -- --headless-then-gui --frames {{frames}}

# ── Linting ──

# Format check
fmt-check:
    cargo fmt --all -- --check

# Auto-format
fmt:
    cargo fmt --all

# Clippy (strict: warnings as errors)
lint:
    cargo clippy --all-targets -- -D warnings

# Clippy (relaxed: show warnings only)
lint-warn:
    cargo clippy --all-targets

# Full lint pass (fmt + clippy)
check: fmt-check lint

# ── Benchmark ──

# Quick benchmark (100 frames, measure GPU time)
bench:
    cargo run --release -- --headless --frames 100 --progress-interval 100

# Longer benchmark (1000 frames)
bench-long:
    cargo run --release -- --headless --frames 1000 --progress-interval 500

# ── CI (same as GitHub Actions) ──

# Full CI check (fmt + clippy + test + build)
ci: fmt-check lint test
    cargo build --release

# ── Presets ──

# Run with a specific preset (loads JSON config)
run-preset preset:
    cargo run --release -- --load presets/{{preset}}.json

# List available presets
list-presets:
    ls presets/
