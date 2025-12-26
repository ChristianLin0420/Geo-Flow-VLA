#!/bin/bash
# ============================================================================
# Geo-Flow VLA Training Launcher
# ============================================================================
# 
# Usage:
#   ./scripts/train.sh phase1 --gpus 0,1        # Train on GPUs 0,1
#   ./scripts/train.sh phase2 --gpus 2,3,4,5    # Train on GPUs 2-5
#   ./scripts/train.sh phase1 --gpus all        # Use all GPUs
#
# ============================================================================

set -e

# Default values
PHASE="phase1"
GPUS="0"
BATCH_SIZE=""
EPOCHS=""
RUN_NAME=""
PROJECT=""
NUM_WORKERS=""
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        phase1|phase2)
            PHASE="$1"
            shift
            ;;
        --gpus|-g)
            GPUS="$2"
            shift 2
            ;;
        --batch-size|-b)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs|-e)
            EPOCHS="$2"
            shift 2
            ;;
        --name|-n)
            RUN_NAME="$2"
            shift 2
            ;;
        --project|-p)
            PROJECT="$2"
            shift 2
            ;;
        --workers|-w)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [phase1|phase2] [options]"
            echo ""
            echo "Options:"
            echo "  --gpus, -g        GPU IDs (comma-separated) or 'all'"
            echo "  --batch-size, -b  Batch size per GPU"
            echo "  --epochs, -e      Number of epochs"
            echo "  --name, -n        WandB run name"
            echo "  --project, -p     WandB project name"
            echo "  --workers, -w     Number of data loader workers (reduce if OOM/shm errors)"
            echo ""
            echo "Examples:"
            echo "  $0 phase1 --gpus 0,1 --epochs 100 --name my_experiment"
            echo "  $0 phase2 --gpus 2,3,4,5 --batch-size 8 --project my-project"
            exit 0
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment if exists
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

cd "$PROJECT_ROOT"

# Determine GPU configuration
if [ "$GPUS" = "all" ]; then
    # Count available GPUs
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    GPU_IDS=$(seq -s, 0 $((NUM_GPUS-1)))
else
    GPU_IDS="$GPUS"
    NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
fi

echo "=============================================="
echo "Geo-Flow VLA Training"
echo "=============================================="
echo "Phase: $PHASE"
echo "GPUs: $GPU_IDS ($NUM_GPUS GPUs)"
[ -n "$BATCH_SIZE" ] && echo "Batch size: $BATCH_SIZE (per GPU)"
[ -n "$EPOCHS" ] && echo "Epochs: $EPOCHS"
[ -n "$PROJECT" ] && echo "WandB project: $PROJECT"
[ -n "$RUN_NAME" ] && echo "WandB run name: $RUN_NAME"
[ -n "$NUM_WORKERS" ] && echo "Workers: $NUM_WORKERS"
echo "=============================================="

# Set CUDA devices - NOTE: torchrun handles device assignment via LOCAL_RANK
# We just need to make the right GPUs visible
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Build command
if [ "$PHASE" = "phase1" ]; then
    MODULE="geo_flow_vla.train.phase1_world_model"
    BATCH_KEY="training.phase1.batch_size"
    EPOCH_KEY="training.phase1.epochs"
else
    MODULE="geo_flow_vla.train.phase2_policy"
    BATCH_KEY="training.phase2.batch_size"
    EPOCH_KEY="training.phase2.epochs"
fi

# Build overrides
OVERRIDES="hardware.distributed=true hardware.num_gpus=$NUM_GPUS"

if [ -n "$BATCH_SIZE" ]; then
    OVERRIDES="$OVERRIDES $BATCH_KEY=$BATCH_SIZE"
fi

if [ -n "$EPOCHS" ]; then
    OVERRIDES="$OVERRIDES $EPOCH_KEY=$EPOCHS"
fi

if [ -n "$NUM_WORKERS" ]; then
    OVERRIDES="$OVERRIDES data.num_workers=$NUM_WORKERS"
fi

if [ -n "$RUN_NAME" ]; then
    OVERRIDES="$OVERRIDES logging.wandb.name=$RUN_NAME"
fi

if [ -n "$PROJECT" ]; then
    OVERRIDES="$OVERRIDES logging.wandb.project=$PROJECT"
fi

# Run training
if [ "$NUM_GPUS" -gt 1 ]; then
    MASTER_PORT=$(shuf -i 29500-29999 -n 1)
    echo "Starting distributed training with $NUM_GPUS GPUs..."
    echo "  Master port: $MASTER_PORT"
    echo "  Module: $MODULE"
    echo "  Overrides: $OVERRIDES"
    
    # Use torchrun for distributed training
    torchrun \
        --standalone \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        -m $MODULE \
        $OVERRIDES \
        $EXTRA_ARGS
else
    echo "Starting single-GPU training..."
    python -m $MODULE \
        $OVERRIDES \
        $EXTRA_ARGS
fi

