#!/bin/bash
# Quick build and run script for EvoLenia

set -e

# Source cargo environment if not already available
if ! command -v cargo &> /dev/null; then
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
fi

MODE=${1:-gui}
FRAMES=${2:-100000}
SNAPSHOT=${3:-/tmp/evolenia_final.snap}
RUNS_DIR=${RUNS_DIR:-/tmp/evolenia_runs}
EXP_FRAMES_DEFAULT=5000000

is_uint() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

echo "🦀 Building EvoLenia v2..."
cargo build --release

case "$MODE" in
    gui)
        echo "🌌 Launching GUI..."
        RUST_LOG=info cargo run --release
        ;;

    headless)
        if is_uint "${2:-}"; then
            FRAMES="$2"
            SNAPSHOT=${3:-/tmp/evolenia_final.snap}
        else
            FRAMES=100000
            SNAPSHOT=${2:-/tmp/evolenia_final.snap}
        fi
        echo "⚡ Running headless for $FRAMES frames..."
        RUST_LOG=info cargo run --release -- --headless --frames "$FRAMES" --save "$SNAPSHOT"
        echo "✅ Snapshot saved to $SNAPSHOT"
        ;;

    headless-view)
        if is_uint "${2:-}"; then
            FRAMES="$2"
            SNAPSHOT=${3:-/tmp/evolenia_final.snap}
        else
            FRAMES=100000
            SNAPSHOT=${2:-/tmp/evolenia_final.snap}
        fi
        echo "⚡ Running headless then opening GUI from final state..."
        RUST_LOG=info cargo run --release -- --headless-then-gui --frames "$FRAMES" --save "$SNAPSHOT"
        ;;

    replay)
        SNAPSHOT=${2:-$SNAPSHOT}
        echo "🖼️  Opening GUI from snapshot: $SNAPSHOT"
        RUST_LOG=info cargo run --release -- --load "$SNAPSHOT"
        ;;

    experiment)
        # Usage: ./run.sh experiment [frames] [label]
        # Example: ./run.sh experiment 5000000 baselineA
        if is_uint "${2:-}"; then
            FRAMES="$2"
            LABEL=${3:-exp}
        else
            FRAMES=$EXP_FRAMES_DEFAULT
            LABEL=${2:-exp}
        fi
        TS=$(date +"%Y%m%d_%H%M%S")
        mkdir -p "$RUNS_DIR"
        SNAPSHOT="$RUNS_DIR/${LABEL}_${TS}.snap"
        LOGFILE="$RUNS_DIR/${LABEL}_${TS}.log"

        echo "🧪 EXPERIMENT MODE"
        echo "   frames:      $FRAMES"
        echo "   snapshot:    $SNAPSHOT"
        echo "   log:         $LOGFILE"
        echo "   runs dir:    $RUNS_DIR"
        echo ""
        echo "⚡ Running headless long batch..."
        RUST_LOG=info cargo run --release -- \
            --headless \
            --frames "$FRAMES" \
            --progress-interval 50000 \
            --save "$SNAPSHOT" \
            2>&1 | tee "$LOGFILE"

        echo ""
        echo "✅ Experiment completed"
        echo "📦 Snapshot: $SNAPSHOT"
        echo "📝 Log:      $LOGFILE"
        echo "▶️  Replay:  ./run.sh replay 0 $SNAPSHOT"
        ;;

    experiment-view)
        # Usage: ./run.sh experiment-view [frames] [label]
        # Example: ./run.sh experiment-view 5000000 baselineA
        if is_uint "${2:-}"; then
            FRAMES="$2"
            LABEL=${3:-exp}
        else
            FRAMES=$EXP_FRAMES_DEFAULT
            LABEL=${2:-exp}
        fi
        TS=$(date +"%Y%m%d_%H%M%S")
        mkdir -p "$RUNS_DIR"
        SNAPSHOT="$RUNS_DIR/${LABEL}_${TS}.snap"
        LOGFILE="$RUNS_DIR/${LABEL}_${TS}.log"

        echo "🧪 EXPERIMENT-VIEW MODE"
        echo "   frames:      $FRAMES"
        echo "   snapshot:    $SNAPSHOT"
        echo "   log:         $LOGFILE"
        echo "   runs dir:    $RUNS_DIR"
        echo ""
        echo "⚡ Running headless long batch..."
        RUST_LOG=info cargo run --release -- \
            --headless \
            --frames "$FRAMES" \
            --progress-interval 50000 \
            --save "$SNAPSHOT" \
            2>&1 | tee "$LOGFILE"

        echo ""
        echo "✅ Experiment completed"
        echo "📦 Snapshot: $SNAPSHOT"
        echo "📝 Log:      $LOGFILE"
        echo "🌌 Opening GUI replay..."
        RUST_LOG=info cargo run --release -- --load "$SNAPSHOT"
        ;;

    *)
        echo "Usage: $0 [gui|headless|headless-view|replay|experiment|experiment-view] [frames] [snapshot_path|label]"
        echo ""
        echo "Examples:"
        echo "  $0 gui"
        echo "  $0 headless 500000 /tmp/runA.snap"
        echo "  $0 headless /tmp/runA.snap"
        echo "  $0 headless-view 200000 /tmp/final.snap"
        echo "  $0 replay /tmp/final.snap"
        echo "  $0 experiment 5000000 baselineA"
        echo "  $0 experiment baselineA"
        echo "  $0 experiment-view 5000000 baselineA"
        echo "  $0 experiment-view baselineA"
        exit 1
        ;;
esac
