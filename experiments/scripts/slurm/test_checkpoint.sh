#!/bin/bash
#SBATCH --job-name=ckpt_test
#SBATCH -o logs/test_checkpoint_%j.out
#SBATCH -e logs/test_checkpoint_%j.err
#SBATCH -A engr-lab-jacobsn
#SBATCH --partition=condo-jacobsn
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4

# ============================================
# Checkpoint Saving Verification Test
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

echo "=============================================="
echo "Checkpoint Saving Test"
echo "=============================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "SATCLIP_ROOT: ${SATCLIP_ROOT}"
echo "SATCLIP_DATA_DIR: ${SATCLIP_DATA_DIR}"
echo "=============================================="

cd ${SATCLIP_ROOT}

echo "Activating conda environment..."
source ${SATCLIP_CONDA_ENV}/bin/activate

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
