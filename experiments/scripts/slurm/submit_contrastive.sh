#!/bin/bash
#SBATCH -J satclip_contrastive          # Job name
#SBATCH -o logs/%x_%j.out               # Output file
#SBATCH -e logs/%x_%j.err               # Error file
#SBATCH -p condo-jacobsn                # Partition (your lab's partition)
#SBATCH --gpus a40:1                    # Request 1 A40 GPU
#SBATCH -A engr-lab-jacobsn             # Account
#SBATCH -t 24:00:00                     # Time limit (24 hours for full training)
#SBATCH --mem=64G                       # Memory (larger for image data)
#SBATCH -n 1                            # Number of tasks
#SBATCH -c 8                            # CPUs per task

# =============================================================================
# SLURM Job Script for SatCLIP Contrastive Training
# =============================================================================
#
# Trains a SatCLIP model with custom learned activation functions using
# the S2-100K Sentinel-2 satellite imagery dataset.
#
# Usage:
#   # Default (SIREN + SH, like original paper):
#   sbatch submit_contrastive.sh
#
#   # With splines instead of SIREN:
#   sbatch submit_contrastive.sh --activation spline
#
#   # With ReLU (often better than SIREN!):
#   sbatch submit_contrastive.sh --activation relu
#
#   # With ViT vision encoder:
#   sbatch submit_contrastive.sh --vision moco_vit16
#
#   # Higher resolution SH:
#   sbatch submit_contrastive.sh --encoding sh_l20
#
#   # Quick test (2 epochs):
#   sbatch submit_contrastive.sh --test
#
# =============================================================================

set -e

# Default values
ACTIVATION="siren"
ENCODING="sh_l10"
VISION="moco_resnet18"
TEST_MODE=false
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --activation)
            ACTIVATION="$2"
            shift 2
            ;;
        --encoding)
            ENCODING="$2"
            shift 2
            ;;
        --vision)
            VISION="$2"
            shift 2
            ;;
        --test)
            TEST_MODE=true
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Create logs directory
mkdir -p logs

# Print job info
echo "=============================================="
echo "SatCLIP Contrastive Training"
echo "=============================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${SLURM_GPUS}"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  Activation: $ACTIVATION"
echo "  Encoding: $ENCODING"
echo "  Vision: $VISION"
echo "  Test mode: $TEST_MODE"
echo "=============================================="

# =============================================================================
# Environment Setup
# =============================================================================

# Conda environment
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate satclip
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    conda activate satclip
fi

# Print environment info
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

# Find project root by looking for experiments directory
if [ -d "${SLURM_SUBMIT_DIR}/experiments" ]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
elif [ -d "${SLURM_SUBMIT_DIR}/../experiments" ]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}/.."
elif [ -d "${SLURM_SUBMIT_DIR}/../../experiments" ]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}/../.."
elif [ -d "${SLURM_SUBMIT_DIR}/../../../experiments" ]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}/../../.."
else
    echo "ERROR: Could not find project root (looking for experiments/ directory)"
    echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR}"
    exit 1
fi

cd "${PROJECT_ROOT}"
echo "Project root: $(pwd)"

# Base config
CMD="python -m experiments.train"
CMD="$CMD --config experiments/configs/experiments/contrastive.yaml"

# Encoding config
if [ -f "experiments/configs/encodings/${ENCODING}.yaml" ]; then
    CMD="$CMD --config experiments/configs/encodings/${ENCODING}.yaml"
else
    echo "Warning: Encoding config not found: experiments/configs/encodings/${ENCODING}.yaml"
fi

# Activation config
if [ -f "experiments/configs/activations/${ACTIVATION}.yaml" ]; then
    CMD="$CMD --config experiments/configs/activations/${ACTIVATION}.yaml"
else
    echo "Warning: Activation config not found: experiments/configs/activations/${ACTIVATION}.yaml"
fi

# Vision encoder
CMD="$CMD --model.vision_encoder=$VISION"

# Test mode
if [ "$TEST_MODE" = true ]; then
    CMD="$CMD --training.max_epochs=2"
    CMD="$CMD --data.batch_size=32"
    CMD="$CMD --training.accumulate_grad_batches=1"
    echo "TEST MODE: Running quick validation (2 epochs, batch_size=32)"
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

# Print completion
echo ""
echo "=============================================="
echo "Job completed at: $(date)"
echo "=============================================="
