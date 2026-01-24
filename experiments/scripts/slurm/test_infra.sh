#!/bin/bash
#SBATCH --job-name=satclip_test
#SBATCH -o logs/test_infra_%j.out
#SBATCH -e logs/test_infra_%j.err
#SBATCH -A engr-lab-jacobsn
#SBATCH --partition=condo-jacobsn
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8

# ============================================
# Infrastructure Test Script
# ============================================
# Verifies: data loading, model creation, checkpointing, logging
# ============================================

set -e

# Load environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for dir in "$SCRIPT_DIR" "$SCRIPT_DIR/.." "$SCRIPT_DIR/../.." "$SCRIPT_DIR/../../.."; do
    if [ -f "$dir/env.sh" ]; then
        source "$dir/env.sh"
        break
    fi
done

# Fallback if env.sh not found
: ${SATCLIP_ROOT:="/engrfs/project/jacobsn/hiqbal/src/satclip"}
: ${SATCLIP_CONDA_ENV:="/engrfs/project/jacobsn/hiqbal/conda/envs/satclip"}
: ${SATCLIP_DATA_DIR:="/engrfs/tmp/jacobsn/hiqbal_satclip"}

echo "=============================================="
echo "SatCLIP Infrastructure Test"
echo "=============================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPUs: ${SLURM_GPUS:-a40:1}"
echo "Start time: $(date)"
echo ""
echo "Paths:"
echo "  SATCLIP_ROOT: ${SATCLIP_ROOT}"
echo "  SATCLIP_DATA_DIR: ${SATCLIP_DATA_DIR}"
echo "  SATCLIP_DATASET_PATH: ${SATCLIP_DATASET_PATH:-${SATCLIP_DATA_DIR}/satclip_hf_preprocessed}"
echo "=============================================="

cd ${SATCLIP_ROOT}

echo "Activating conda environment..."
source ${SATCLIP_CONDA_ENV}/bin/activate

echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q "True"; then
    echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo ""

echo "Running infrastructure test..."
echo "=============================================="
python -m experiments.scripts.test_training_infra

echo ""
echo "=============================================="
echo "Test completed at: $(date)"
echo "=============================================="
