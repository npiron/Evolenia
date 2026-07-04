#!/bin/bash
# Performance profiles for EvoLenia

set -e

# Source cargo environment if not already available
if ! command -v cargo &> /dev/null; then
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
fi

PROFILE=${1:-balanced}

case $PROFILE in
  fast)
    echo "🚀 FAST MODE: 512×512, max_r=6, DT=0.1 (~240 FPS)"
    sed -i.bak 's/pub const WORLD_WIDTH: u32 = [0-9]*/pub const WORLD_WIDTH: u32 = 512/' src/world/constants.rs
    sed -i.bak 's/pub const WORLD_HEIGHT: u32 = [0-9]*/pub const WORLD_HEIGHT: u32 = 512/' src/world/constants.rs
    sed -i.bak 's/pub const DT: f32 = [0-9.]*/pub const DT: f32 = 0.1/' src/world/constants.rs
    sed -i.bak 's/let max_r   = [0-9]*/let max_r   = 6/' src/shaders/compute_evolution.wgsl
    ;;
    
  balanced)
    echo "⚖️  BALANCED MODE: 1024×1024, max_r=9, DT=0.05 (~60 FPS)"
    sed -i.bak 's/pub const WORLD_WIDTH: u32 = [0-9]*/pub const WORLD_WIDTH: u32 = 1024/' src/world/constants.rs
    sed -i.bak 's/pub const WORLD_HEIGHT: u32 = [0-9]*/pub const WORLD_HEIGHT: u32 = 1024/' src/world/constants.rs
    sed -i.bak 's/pub const DT: f32 = [0-9.]*/pub const DT: f32 = 0.05/' src/world/constants.rs
    sed -i.bak 's/let max_r   = [0-9]*/let max_r   = 9/' src/shaders/compute_evolution.wgsl
    ;;
    
  quality)
    echo "💎 QUALITY MODE: 2048×2048, max_r=12, DT=0.03 (~15 FPS, requires powerful GPU)"
    sed -i.bak 's/pub const WORLD_WIDTH: u32 = [0-9]*/pub const WORLD_WIDTH: u32 = 2048/' src/world/constants.rs
    sed -i.bak 's/pub const WORLD_HEIGHT: u32 = [0-9]*/pub const WORLD_HEIGHT: u32 = 2048/' src/world/constants.rs
    sed -i.bak 's/pub const DT: f32 = [0-9.]*/pub const DT: f32 = 0.03/' src/world/constants.rs
    sed -i.bak 's/let max_r   = [0-9]*/let max_r   = 12/' src/shaders/compute_evolution.wgsl
    ;;
    
  *)
    echo "Usage: $0 [fast|balanced|quality]"
    echo ""
    echo "Profiles:"
    echo "  fast     - 512×512, optimized for development (4× faster)"
    echo "  balanced - 1024×1024, default settings (recommended)"
    echo "  quality  - 2048×2048, maximum detail (slow)"
    exit 1
    ;;
esac

# Cleanup backup files
rm -f src/world/constants.rs.bak src/shaders/compute_evolution.wgsl.bak

echo "✅ Profile applied. Rebuilding..."
cargo build --release

echo "🎮 Ready to run: cargo run --release"
