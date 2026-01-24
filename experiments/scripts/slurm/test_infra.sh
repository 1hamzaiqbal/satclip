#!/bin/bash
#SBATCH --job-name=satclip_test
#SBATCH --output=/engrfs/project/jacobsn/hiqbal/src/satclip/logs/test_infra_%j.out
#SBATCH --error=/engrfs/project/jacobsn/hiqbal/src/satclip/logs/test_infra_%j.err
#SBATCH -A engr-lab-jacobsn
#SBATCH --partition=condo-jacobsn
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8

# ============================================
# Infrastructure Test Script
# ============================================
# This script runs a quick test to verify:
# 1. Data loading works
# 2. Model creation works
# 3. Checkpointing works
# 4. Logging works
#
# Usage:
#   sbatch experiments/scripts/slurm/test_infra.sh
# ============================================

set -e

echo "=============================================="
echo "SatCLIP Infrastructure Test"
echo "=============================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPUs: ${SLURM_JOB_GPUS:-a40:1}"
echo "Start time: $(date)"
echo "=============================================="

# Key paths (KEEP THESE UPDATED)
PROJECT_ROOT="/engrfs/project/jacobsn/hiqbal/src/satclip"
CONDA_ENV="/engrfs/project/jacobsn/hiqbal/conda/envs/satclip"
TEMP_DIR="/engrfs/tmp/jacobsn/hiqbal_satclip"

# Print paths for debugging
echo ""
echo "Paths:"
echo "  Project root: ${PROJECT_ROOT}"
echo "  Conda env: ${CONDA_ENV}"
echo "  Temp directory: ${TEMP_DIR}"
echo "  Dataset: ${TEMP_DIR}/satclip_hf_preprocessed"
echo "  Output: ${TEMP_DIR}/test_output"
echo ""

# Environment setup
cd ${PROJECT_ROOT}

echo "Activating conda environment..."
source ${CONDA_ENV}/bin/activate

echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q "True"; then
    echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo ""

# Run test
echo "Running infrastructure test..."
echo "=============================================="
python -m experiments.scripts.test_training_infra

echo ""
echo "=============================================="
echo "Test completed at: $(date)"
echo "=============================================="
