#!/bin/bash
# Quick build and run script for EvoLenia

set -e

# Source cargo environment if not already available
if ! command -v cargo &> /dev/null; then
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
fi

echo "🦀 Building EvoLenia v2..."
cargo build --release

echo "🌌 Launching simulation..."
RUST_LOG=info cargo run --release
