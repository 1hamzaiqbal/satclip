#!/bin/bash
# ============================================
# SatCLIP Environment Configuration
# ============================================
# Source this file to set up environment variables.
# All scripts reference these variables instead of hardcoded paths.
#
# Usage:
#   source env.sh        # Auto-detects HPC vs local
#   source env.sh hpc    # Force HPC paths
#   source env.sh local  # Force local paths
# ============================================

# Detect environment or use argument
if [[ "$1" == "hpc" ]] || [[ -d "/engrfs" ]]; then
    SATCLIP_ENV="hpc"
elif [[ "$1" == "local" ]] || [[ -d "/Users" ]]; then
    SATCLIP_ENV="local"
else
    SATCLIP_ENV="unknown"
fi

# ============================================
# HPC Environment (WUSTL)
# ============================================
if [[ "$SATCLIP_ENV" == "hpc" ]]; then
    # Project paths
    export SATCLIP_ROOT="/engrfs/project/jacobsn/hiqbal/src/satclip"
    export SATCLIP_CONDA_ENV="/engrfs/project/jacobsn/hiqbal/conda/envs/satclip"

    # Data paths (temp pool - unlimited quota)
    export SATCLIP_DATA_DIR="/engrfs/tmp/jacobsn/hiqbal_satclip"
    export SATCLIP_DATASET_PATH="${SATCLIP_DATA_DIR}/satclip_hf_preprocessed"
    export SATCLIP_LOGS_DIR="${SATCLIP_DATA_DIR}/logs"
    export SATCLIP_HF_CACHE="${SATCLIP_DATA_DIR}/hf_cache"

    # SLURM settings
    export SATCLIP_SLURM_ACCOUNT="engr-lab-jacobsn"
    export SATCLIP_SLURM_PARTITION="condo-jacobsn"
    export SATCLIP_SLURM_GPU="a40:1"

    # Job output directory (project folder for SLURM logs)
    export SATCLIP_SLURM_LOGS="${SATCLIP_ROOT}/logs"

# ============================================
# Local Environment (Mac/Linux dev)
# ============================================
elif [[ "$SATCLIP_ENV" == "local" ]]; then
    # Project paths
    export SATCLIP_ROOT="/Users/hamzaiqbal/grad/learned_activation/satclip"
    export SATCLIP_CONDA_ENV=""  # Use system Python or activated conda

    # Data paths
    export SATCLIP_DATA_DIR="${SATCLIP_ROOT}/data"
    export SATCLIP_DATASET_PATH="${SATCLIP_DATA_DIR}/satclip_hf_preprocessed"
    export SATCLIP_LOGS_DIR="${SATCLIP_ROOT}/logs"
    export SATCLIP_HF_CACHE="${SATCLIP_DATA_DIR}/hf_cache"

    # No SLURM locally
    export SATCLIP_SLURM_ACCOUNT=""
    export SATCLIP_SLURM_PARTITION=""
    export SATCLIP_SLURM_GPU=""
    export SATCLIP_SLURM_LOGS=""

else
    echo "Warning: Unknown environment. Set paths manually or use: source env.sh [hpc|local]"
fi

# ============================================
# Common settings (both environments)
# ============================================
export SATCLIP_CONFIGS="${SATCLIP_ROOT}/experiments/configs"
export SATCLIP_SCRIPTS="${SATCLIP_ROOT}/experiments/scripts"

# Print summary if sourced interactively
if [[ "$-" == *i* ]] || [[ "$SATCLIP_VERBOSE" == "1" ]]; then
    echo "SatCLIP Environment: ${SATCLIP_ENV}"
    echo "  SATCLIP_ROOT:         ${SATCLIP_ROOT}"
    echo "  SATCLIP_DATA_DIR:     ${SATCLIP_DATA_DIR}"
    echo "  SATCLIP_DATASET_PATH: ${SATCLIP_DATASET_PATH}"
    echo "  SATCLIP_LOGS_DIR:     ${SATCLIP_LOGS_DIR}"
fi
