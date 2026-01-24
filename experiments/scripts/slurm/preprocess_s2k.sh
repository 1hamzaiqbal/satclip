#!/bin/bash
#SBATCH -J preprocess_s2k             # Job name
#SBATCH -o logs/%x_%j.out             # Output file
#SBATCH -e logs/%x_%j.err             # Error file
#SBATCH -p condo-jacobsn              # Partition
#SBATCH -A engr-lab-jacobsn           # Account
#SBATCH -t 12:00:00                   # Time limit (12 hours - extraction takes time)
#SBATCH --mem=64G                     # Memory (need more for dataset building)
#SBATCH -n 1                          # Number of tasks
#SBATCH -c 16                         # CPUs per task (for parallel extraction)

# =============================================================================
# SLURM Job Script for S2-100K Dataset Preprocessing
# =============================================================================
#
# This script performs the full preprocessing pipeline for the S2-100K dataset:
# 1. Extract tar.xz files (optional, if not already extracted)
# 2. Build HuggingFace Arrow dataset from TIF files
# 3. Preprocess (normalize, pad to 13 channels, optional crop)
#
# The preprocessed dataset is MUCH faster for training (2-3x speedup).
#
# Usage:
#   sbatch preprocess_s2k.sh                          # Full pipeline
#   sbatch preprocess_s2k.sh --skip-extract           # Skip extraction
#   sbatch preprocess_s2k.sh --skip-build             # Skip building (preprocess only)
#   sbatch preprocess_s2k.sh --crop 224               # Also center crop to 224x224
#
# Required input structure:
#   DATA_DIR/
#   ├── index.csv                     # CSV with filename, lon, lat columns
#   └── images/
#       └── data_*.tar.xz             # Tar archives with TIF files
#
# =============================================================================

# Exit on error
set -e

# Create logs directory
mkdir -p logs

# Print job info
echo "=============================================="
echo "S2-100K Dataset Preprocessing Pipeline"
echo "=============================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}"
echo "Start time: $(date)"
echo "=============================================="

# =============================================================================
# Configuration - UPDATE THESE PATHS FOR YOUR ENVIRONMENT
# =============================================================================

# Path to raw data (tar.xz files and index.csv)
# UPDATE THESE PATHS FOR YOUR ENVIRONMENT
DATA_DIR="${DATA_DIR:-/engrfs/project/jacobsn/hiqbal/data_raw/satclip}"

# Output paths (on fast storage, e.g., scratch or project space)
OUTPUT_BASE="${OUTPUT_BASE:-/engrfs/project/jacobsn/hiqbal/data_processed/satclip}"
TIF_DIR="${TIF_DIR:-${OUTPUT_BASE}/tif}"
HF_OUTPUT="${HF_OUTPUT:-${OUTPUT_BASE}/satclip_hf}"
PREPROCESSED_OUTPUT="${PREPROCESSED_OUTPUT:-${OUTPUT_BASE}/satclip_hf_preprocessed}"

# Processing options
SKIP_EXTRACT="${SKIP_EXTRACT:-false}"
SKIP_BUILD="${SKIP_BUILD:-false}"
CROP_SIZE="${CROP_SIZE:-}"  # Empty = no crop, or e.g., "224"
NUM_WORKERS="${NUM_WORKERS:-8}"

# =============================================================================
# Parse Command Line Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            HF_OUTPUT="$2"
            shift 2
            ;;
        --skip-extract)
            SKIP_EXTRACT=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --crop)
            CROP_SIZE="$2"
            shift 2
            ;;
        --workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Environment Setup
# =============================================================================

# Option 1: Conda environment
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
echo "  NumPy: $(python -c 'import numpy; print(numpy.__version__)')"
echo ""

# Print configuration
echo "Configuration:"
echo "  DATA_DIR: ${DATA_DIR}"
echo "  TIF_DIR: ${TIF_DIR}"
echo "  HF_OUTPUT: ${HF_OUTPUT}"
echo "  PREPROCESSED_OUTPUT: ${PREPROCESSED_OUTPUT}"
echo "  SKIP_EXTRACT: ${SKIP_EXTRACT}"
echo "  SKIP_BUILD: ${SKIP_BUILD}"
echo "  CROP_SIZE: ${CROP_SIZE:-'none (keep original)'}"
echo "  NUM_WORKERS: ${NUM_WORKERS}"
echo ""

# Change to project directory
# Find project root by looking for experiments directory
# Works whether submitting from repo root, scripts/, or slurm/ subdirectory
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
SCRIPTS_DIR="experiments/scripts/preprocessing"

# =============================================================================
# Step 1: Extract tar.xz files
# =============================================================================

if [ "$SKIP_EXTRACT" = false ]; then
    echo "=============================================="
    echo "Step 1: Extracting tar.xz files"
    echo "=============================================="

    python "${SCRIPTS_DIR}/build_hf_dataset.py" \
        --data-dir "${DATA_DIR}" \
        --tif-dir "${TIF_DIR}" \
        --workers "${NUM_WORKERS}" \
        --extract

    echo "Extraction complete!"
else
    echo "Skipping extraction (--skip-extract)"
fi

# =============================================================================
# Step 2: Build HuggingFace dataset
# =============================================================================

if [ "$SKIP_BUILD" = false ]; then
    echo ""
    echo "=============================================="
    echo "Step 2: Building HuggingFace Arrow dataset"
    echo "=============================================="

    python "${SCRIPTS_DIR}/build_hf_dataset.py" \
        --data-dir "${DATA_DIR}" \
        --tif-dir "${TIF_DIR}" \
        --output "${HF_OUTPUT}" \
        --workers "${NUM_WORKERS}" \
        --build

    echo "HuggingFace dataset built!"
else
    echo "Skipping build (--skip-build)"
fi

# =============================================================================
# Step 3: Preprocess dataset
# =============================================================================

echo ""
echo "=============================================="
echo "Step 3: Preprocessing dataset"
echo "=============================================="

PREPROCESS_ARGS="--input ${HF_OUTPUT} --output ${PREPROCESSED_OUTPUT} --num-proc ${NUM_WORKERS}"

if [ -n "${CROP_SIZE}" ]; then
    PREPROCESS_ARGS="${PREPROCESS_ARGS} --crop ${CROP_SIZE}"
fi

python "${SCRIPTS_DIR}/preprocess_hf_dataset.py" ${PREPROCESS_ARGS}

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================================="
echo "PREPROCESSING COMPLETE"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Output locations:"
echo "  Raw TIF files: ${TIF_DIR}"
echo "  HuggingFace dataset: ${HF_OUTPUT}"
echo "  Preprocessed dataset: ${PREPROCESSED_OUTPUT}"
echo ""
echo "To use in training, update your config:"
echo ""
echo "data:"
echo "  use_hf_dataset: true"
echo "  hf_dataset_path: \"${PREPROCESSED_OUTPUT}\""
echo "  pad_to_13_channels: false  # Already padded"
echo "  preprocessed: true          # Already normalized"
echo ""
echo "Or run training with:"
echo "  python -m experiments.train \\"
echo "    --config experiments/configs/experiments/contrastive.yaml \\"
echo "    --data.hf_dataset_path=${PREPROCESSED_OUTPUT}"
echo ""
