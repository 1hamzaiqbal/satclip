#!/bin/bash
#SBATCH -J satclip_contrastive
#SBATCH -o logs/satclip_contrastive_%j.out
#SBATCH -e logs/satclip_contrastive_%j.err
#SBATCH -p condo-jacobsn
#SBATCH --gpus a40:1
#SBATCH -A engr-lab-jacobsn
#SBATCH -t 24:00:00
#SBATCH --mem=192G
#SBATCH -n 1
#SBATCH -c 16

# =============================================================================
# SLURM Job Script for SatCLIP Contrastive Training
# =============================================================================
#
# Usage:
#   sbatch submit_contrastive.sh                          # Default (SIREN + SH)
#   sbatch submit_contrastive.sh --activation spline      # Spline activation
#   sbatch submit_contrastive.sh --short --activation spline  # 50 epoch monitoring
#   sbatch submit_contrastive.sh --test                   # Quick 2 epoch test
#
# =============================================================================

set -e

# =============================================================================
# Load Environment
# =============================================================================

# Find and source env.sh (search up from script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE=""

for dir in "$SCRIPT_DIR" "$SCRIPT_DIR/.." "$SCRIPT_DIR/../.." "$SCRIPT_DIR/../../.."; do
    if [ -f "$dir/env.sh" ]; then
        ENV_FILE="$dir/env.sh"
        break
    fi
done

if [ -n "$ENV_FILE" ]; then
    source "$ENV_FILE"
    echo "Loaded environment from: $ENV_FILE"
else
    echo "Warning: env.sh not found, using defaults"
    # Fallback defaults
    export SATCLIP_ROOT="/engrfs/project/jacobsn/hiqbal/src/satclip"
    export SATCLIP_DATA_DIR="/engrfs/tmp/jacobsn/hiqbal_satclip"
    export SATCLIP_CONDA_ENV="/engrfs/project/jacobsn/hiqbal/conda/envs/satclip"
fi

# =============================================================================
# Parse Arguments
# =============================================================================

ACTIVATION="siren"
ENCODING="sh_l10"
VISION="moco_resnet18"
TEST_MODE=false
SHORT_MODE=false
BATCH_SIZE=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --activation) ACTIVATION="$2"; shift 2 ;;
        --encoding) ENCODING="$2"; shift 2 ;;
        --vision) VISION="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --test) TEST_MODE=true; shift ;;
        --short) SHORT_MODE=true; shift ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# =============================================================================
# Print Configuration
# =============================================================================

echo "=============================================="
echo "SatCLIP Contrastive Training"
echo "=============================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPUs: ${SLURM_GPUS:-a40:1}"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  Activation: $ACTIVATION"
echo "  Encoding: $ENCODING"
echo "  Vision: $VISION"
echo "  Test mode: $TEST_MODE"
echo "  Short mode: $SHORT_MODE"
echo "=============================================="
echo ""
echo "Paths:"
echo "  SATCLIP_ROOT: ${SATCLIP_ROOT}"
echo "  SATCLIP_DATA_DIR: ${SATCLIP_DATA_DIR}"
echo "  SATCLIP_LOGS_DIR: ${SATCLIP_LOGS_DIR:-${SATCLIP_DATA_DIR}/logs}"
echo "=============================================="

# =============================================================================
# Environment Setup
# =============================================================================

# Activate conda environment
if [ -n "$SATCLIP_CONDA_ENV" ] && [ -f "${SATCLIP_CONDA_ENV}/bin/activate" ]; then
    source "${SATCLIP_CONDA_ENV}/bin/activate"
fi

# Set HuggingFace cache
export HF_DATASETS_CACHE="${SATCLIP_DATA_DIR}/hf_cache"
export HF_HOME="${SATCLIP_DATA_DIR}/hf_home"

echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  Device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"
echo ""

# =============================================================================
# Build Training Command
# =============================================================================

cd "${SATCLIP_ROOT}"
echo "Project root: ${SATCLIP_ROOT}"

# Build command using environment paths
CMD="python -m experiments.train"
CMD="$CMD --config experiments/configs/experiments/contrastive_multispectral.yaml"

# Encoding config
if [ -f "experiments/configs/encodings/${ENCODING}.yaml" ]; then
    CMD="$CMD --config experiments/configs/encodings/${ENCODING}.yaml"
fi

# Activation config
if [ -f "experiments/configs/activations/${ACTIVATION}.yaml" ]; then
    CMD="$CMD --config experiments/configs/activations/${ACTIVATION}.yaml"
fi

# Vision encoder
CMD="$CMD --model.vision_encoder=$VISION"

# Test mode
if [ "$TEST_MODE" = true ]; then
    CMD="$CMD --training.max_epochs=2 --data.batch_size=32 --training.accumulate_grad_batches=1"
    echo "TEST MODE: Running 2 epochs with batch_size=32"
fi

# Short mode (with reduced batch size for stability)
if [ "$SHORT_MODE" = true ]; then
    CMD="$CMD --config experiments/configs/experiments/contrastive_short.yaml"
    # Default to smaller batch if not specified
    if [ -z "$BATCH_SIZE" ]; then
        BATCH_SIZE=256
    fi
    echo "SHORT MODE: Running 50 epochs for monitoring"
fi

# Batch size override
if [ -n "$BATCH_SIZE" ]; then
    CMD="$CMD --data.batch_size=$BATCH_SIZE"
    echo "BATCH SIZE: $BATCH_SIZE"
fi

# Extra args
CMD="$CMD $EXTRA_ARGS"

# =============================================================================
# Run Training
# =============================================================================

echo "Running command:"
echo "  $CMD"
echo ""

eval $CMD

echo ""
echo "=============================================="
echo "Job completed at: $(date)"
echo "=============================================="
