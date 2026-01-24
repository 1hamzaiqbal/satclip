#!/bin/bash
#SBATCH --job-name=ckpt_test
#SBATCH --output=/engrfs/project/jacobsn/hiqbal/src/satclip/logs/test_checkpoint_%j.out
#SBATCH --error=/engrfs/project/jacobsn/hiqbal/src/satclip/logs/test_checkpoint_%j.err
#SBATCH -A engr-lab-jacobsn
#SBATCH --partition=condo-jacobsn
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4

# ============================================
# Checkpoint Saving Verification Test
# ============================================
# Runs 5 quick epochs with checkpoints every 2 epochs
# to verify checkpoint saving works correctly.
# ============================================

set -e

echo "=============================================="
echo "Checkpoint Saving Test"
echo "=============================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=============================================="

PROJECT_ROOT="/engrfs/project/jacobsn/hiqbal/src/satclip"
CONDA_ENV="/engrfs/project/jacobsn/hiqbal/conda/envs/satclip"

cd ${PROJECT_ROOT}

echo "Activating conda environment..."
source ${CONDA_ENV}/bin/activate

echo ""
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

echo "Running checkpoint verification test..."
echo "=============================================="
python -m experiments.scripts.test_checkpoint_save

echo ""
echo "=============================================="
echo "Test completed at: $(date)"
echo "=============================================="
