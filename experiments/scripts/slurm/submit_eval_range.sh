#!/bin/bash
#SBATCH -J satclip_eval_range
#SBATCH -o logs/satclip_eval_range_%j.out
#SBATCH -e logs/satclip_eval_range_%j.err
#SBATCH -p condo-jacobsn
#SBATCH --gpus a40:1
#SBATCH -A engr-lab-jacobsn
#SBATCH -t 2:00:00
#SBATCH --mem=64G
#SBATCH -n 1
#SBATCH -c 8

# =============================================================================
# SLURM Job Script for RANGE Evaluation of SatCLIP Models
# =============================================================================
#
# Usage:
#   sbatch submit_eval_range.sh --checkpoint /path/to/checkpoint.ckpt
#   sbatch submit_eval_range.sh --checkpoint /path/to/checkpoint.ckpt --tasks biome country
#   sbatch submit_eval_range.sh --checkpoint /path/to/checkpoint.ckpt --output_dir /path/to/output
#
# =============================================================================

set -e

# =============================================================================
# Load Environment
# =============================================================================

# Find and source env.sh
ENV_FILE=""

# Try 1: Known HPC path (most reliable for SLURM jobs)
if [ -f "/engrfs/project/jacobsn/hiqbal/src/satclip/env.sh" ]; then
    ENV_FILE="/engrfs/project/jacobsn/hiqbal/src/satclip/env.sh"
fi

# Try 2: Search from script location
if [ -z "$ENV_FILE" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
    for dir in "$SCRIPT_DIR" "$SCRIPT_DIR/.." "$SCRIPT_DIR/../.." "$SCRIPT_DIR/../../.."; do
        if [ -f "$dir/env.sh" ]; then
            ENV_FILE="$dir/env.sh"
            break
        fi
    done
fi

if [ -n "$ENV_FILE" ]; then
    source "$ENV_FILE"
    echo "Loaded environment from: $ENV_FILE"
else
    echo "Warning: env.sh not found, using defaults"
    export SATCLIP_ROOT="/engrfs/project/jacobsn/hiqbal/src/satclip"
    export SATCLIP_DATA_DIR="/engrfs/tmp/jacobsn/hiqbal_satclip"
    export SATCLIP_CONDA_ENV="/engrfs/project/jacobsn/hiqbal/conda/envs/satclip"
fi

# =============================================================================
# Parse Arguments
# =============================================================================

CHECKPOINT=""
OUTPUT_DIR=""
TASKS=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --tasks)
            # Collect all task arguments until next flag
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                TASKS="$TASKS $1"
                shift
            done
            ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# Validate checkpoint
if [ -z "$CHECKPOINT" ]; then
    echo "Error: --checkpoint is required"
    echo "Usage: sbatch submit_eval_range.sh --checkpoint /path/to/checkpoint.ckpt"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# Default output directory
if [ -z "$OUTPUT_DIR" ]; then
    CKPT_DIR=$(dirname "$CHECKPOINT")
    OUTPUT_DIR="${CKPT_DIR}/eval_range"
fi

# =============================================================================
# Print Configuration
# =============================================================================

echo "=============================================="
echo "RANGE Evaluation for SatCLIP"
echo "=============================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPUs: ${SLURM_GPUS:-a40:1}"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output dir: $OUTPUT_DIR"
echo "  Tasks: ${TASKS:-all}"
echo "=============================================="

# =============================================================================
# Environment Setup
# =============================================================================

# Activate conda environment
if [ -n "$SATCLIP_CONDA_ENV" ] && [ -f "${SATCLIP_CONDA_ENV}/bin/activate" ]; then
    source "${SATCLIP_CONDA_ENV}/bin/activate"
fi

# Set data directory
export SATCLIP_RANGE_DATA="/engrfs/project/jacobsn/hiqbal/data_raw/datasets/eval_range_datasets"

echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  SATCLIP_RANGE_DATA: $SATCLIP_RANGE_DATA"
echo "  CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  Device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"
echo ""

# =============================================================================
# Run Evaluation
# =============================================================================

cd "${SATCLIP_ROOT}"

CMD="python -m experiments.eval_range"
CMD="$CMD --model_path $CHECKPOINT"
CMD="$CMD --output_dir $OUTPUT_DIR"
CMD="$CMD --device cuda"
CMD="$CMD --batch_size 512"
CMD="$CMD --num_workers 4"

if [ -n "$TASKS" ]; then
    CMD="$CMD --tasks $TASKS"
fi

CMD="$CMD $EXTRA_ARGS"

echo "Running command:"
echo "  $CMD"
echo ""

eval $CMD

echo ""
echo "=============================================="
echo "Evaluation completed at: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
