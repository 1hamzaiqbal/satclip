#!/bin/bash
#SBATCH -J eval_range                 # Job name
#SBATCH -o logs/%x_%j.out             # Output file
#SBATCH -e logs/%x_%j.err             # Error file
#SBATCH -p condo-jacobsn              # Partition
#SBATCH --gpus a40:1                  # Request 1 GPU
#SBATCH -A engr-lab-jacobsn           # Account
#SBATCH -t 2:00:00                    # Time limit (2 hours - eval is fast)
#SBATCH --mem=32G                     # Memory
#SBATCH -n 1                          # Number of tasks
#SBATCH -c 8                          # CPUs per task

# =============================================================================
# SLURM Job Script for RANGE Evaluation
# =============================================================================
#
# Evaluates a trained location encoder on the RANGE benchmark tasks:
# - Classification: biome, ecoregion, country, ocean
# - Regression: temperature, housing, elevation, population, ERA5
# - Synthetic: checker_100, checker_200
#
# Usage:
#   sbatch eval_range.sh --model_path /path/to/checkpoint.pt
#   sbatch eval_range.sh --model_path /path/to/checkpoint.pt --tasks biome,elevation
#   sbatch eval_range.sh --model_path /path/to/checkpoint.pt --output_dir /path/to/results
#
# =============================================================================

# Exit on error
set -e

# Create logs directory
mkdir -p logs

# Print job info
echo "=============================================="
echo "RANGE Evaluation"
echo "=============================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${SLURM_GPUS}"
echo "Start time: $(date)"
echo "=============================================="

# =============================================================================
# Configuration
# =============================================================================

# Path to RANGE evaluation data
DATA_DIR="${RANGE_DATA_DIR:-/projects/bdbk/shared/range_eval_data}"

# Default tasks (all available)
DEFAULT_TASKS="biome,ecoregion,country,ocean,temperature,housing,elevation,population,checker_100,checker_200"

# Parse arguments
MODEL_PATH=""
OUTPUT_DIR=""
TASKS="${DEFAULT_TASKS}"
DEVICE="cuda"
BATCH_SIZE=256

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --data_dir|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --output_dir|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --batch_size|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "${MODEL_PATH}" ]; then
    echo "Error: --model_path is required"
    echo "Usage: sbatch eval_range.sh --model_path /path/to/checkpoint.pt"
    exit 1
fi

# Set default output directory
if [ -z "${OUTPUT_DIR}" ]; then
    MODEL_DIR=$(dirname "${MODEL_PATH}")
    OUTPUT_DIR="${MODEL_DIR}/eval_results"
fi

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
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Print configuration
echo "Configuration:"
echo "  MODEL_PATH: ${MODEL_PATH}"
echo "  DATA_DIR: ${DATA_DIR}"
echo "  TASKS: ${TASKS}"
echo "  OUTPUT_DIR: ${OUTPUT_DIR}"
echo "  DEVICE: ${DEVICE}"
echo "  BATCH_SIZE: ${BATCH_SIZE}"
echo ""

# =============================================================================
# Run Evaluation
# =============================================================================

# Change to project directory
cd "${SLURM_SUBMIT_DIR}/../.."

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Convert comma-separated tasks to space-separated
TASK_ARGS=$(echo "${TASKS}" | tr ',' ' ')

echo "Running RANGE evaluation..."
python experiments/eval_range.py \
    --model_path "${MODEL_PATH}" \
    --data_dir "${DATA_DIR}" \
    --tasks ${TASK_ARGS} \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --output_dir "${OUTPUT_DIR}"

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================================="
echo "EVALUATION COMPLETE"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo ""

# Print results if JSON exists
RESULTS_FILE="${OUTPUT_DIR}/eval_results.json"
if [ -f "${RESULTS_FILE}" ]; then
    echo "Results:"
    cat "${RESULTS_FILE}"
fi
