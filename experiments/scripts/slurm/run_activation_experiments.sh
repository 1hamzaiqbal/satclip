#!/bin/bash
# =============================================================================
# Run Activation Function Experiments
# =============================================================================
#
# Submits multiple SatCLIP contrastive training jobs with different activation
# functions for comparison:
#   1. Spline (ReLU init) - Primary experiment
#   2. ReLU - Baseline (often better than SIREN!)
#   3. SIREN - Original SatCLIP baseline
#
# Usage:
#   # Submit all three experiments:
#   ./run_activation_experiments.sh
#
#   # Submit with test mode (2 epochs each):
#   ./run_activation_experiments.sh --test
#
#   # Submit only spline:
#   ./run_activation_experiments.sh --only spline
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_MODE=""
ONLY_ACTIVATION=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE="--test"
            shift
            ;;
        --only)
            ONLY_ACTIVATION="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Activation Function Experiments"
echo "=============================================="
echo "Script dir: $SCRIPT_DIR"
echo "Test mode: ${TEST_MODE:-no}"
echo "Only activation: ${ONLY_ACTIVATION:-all}"
echo ""

# Function to submit a job
submit_job() {
    local activation=$1
    local encoding=${2:-sh_l10}

    echo "Submitting: $activation + $encoding"

    JOB_ID=$(sbatch --parsable \
        --job-name="satclip_${activation}" \
        "${SCRIPT_DIR}/submit_contrastive.sh" \
        --activation "$activation" \
        --encoding "$encoding" \
        $TEST_MODE)

    echo "  -> Job ID: $JOB_ID"
    echo ""
}

# Submit experiments
if [ -z "$ONLY_ACTIVATION" ] || [ "$ONLY_ACTIVATION" = "spline" ]; then
    echo "=== Experiment 1: Spline (ReLU init) ==="
    submit_job "spline" "sh_l10"
fi

if [ -z "$ONLY_ACTIVATION" ] || [ "$ONLY_ACTIVATION" = "relu" ]; then
    echo "=== Experiment 2: ReLU Baseline ==="
    submit_job "relu" "sh_l10"
fi

if [ -z "$ONLY_ACTIVATION" ] || [ "$ONLY_ACTIVATION" = "siren" ]; then
    echo "=== Experiment 3: SIREN (Original SatCLIP) ==="
    submit_job "siren" "sh_l10"
fi

echo "=============================================="
echo "All jobs submitted!"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo ""
echo "View TensorBoard logs:"
echo "  tensorboard --logdir=/engrfs/project/jacobsn/hiqbal/logs/satclip"
echo "=============================================="
