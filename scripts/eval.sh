#!/bin/bash
# Geo-Flow-VLA Unified Evaluation Script
#
# Usage:
#   bash scripts/eval.sh libero ./checkpoints/libero/full [OPTIONS]
#   bash scripts/eval.sh rlbench ./checkpoints/rlbench/all [OPTIONS]
#   bash scripts/eval.sh calvin ./checkpoints/calvin/abc --calvin_root ./data/calvin [OPTIONS]

set -e

BENCHMARK=${1:-"libero"}
CHECKPOINT=${2:-"./checkpoints/libero/full"}
shift 2 || true

echo "========================================"
echo "Geo-Flow-VLA Evaluation"
echo "========================================"
echo "Benchmark: ${BENCHMARK}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Additional args: $@"
echo "========================================"

case $BENCHMARK in
    libero)
        python -m geo_flow_vla.eval.eval_libero \
            --checkpoint "$CHECKPOINT" \
            --suite libero_10 \
            --n_rollouts 50 \
            --max_steps 400 \
            "$@"
        ;;
    rlbench)
        python -m geo_flow_vla.eval.eval_rlbench \
            --checkpoint "$CHECKPOINT" \
            --category all \
            --n_rollouts 25 \
            --max_steps 200 \
            "$@"
        ;;
    calvin)
        python -m geo_flow_vla.eval.eval_calvin \
            --checkpoint "$CHECKPOINT" \
            --split D \
            --n_chains 1000 \
            "$@"
        ;;
    *)
        echo "Unknown benchmark: $BENCHMARK"
        echo ""
        echo "Usage: $0 {libero|rlbench|calvin} CHECKPOINT [OPTIONS]"
        echo ""
        echo "Benchmarks:"
        echo "  libero   - LIBERO manipulation benchmark (MuJoCo)"
        echo "  rlbench  - RLBench 18-task benchmark (CoppeliaSim)"
        echo "  calvin   - CALVIN long-horizon benchmark (PyBullet)"
        echo ""
        echo "Examples:"
        echo "  $0 libero ./checkpoints/libero/full --suite libero_10 --n_rollouts 50"
        echo "  $0 rlbench ./checkpoints/rlbench/all --tasks reach_target push_buttons"
        echo "  $0 calvin ./checkpoints/calvin/abc --calvin_root ./data/calvin --split D"
        exit 1
        ;;
esac

echo ""
echo "✓ Evaluation complete"
