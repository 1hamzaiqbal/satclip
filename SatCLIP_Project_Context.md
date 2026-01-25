# SatCLIP Project Context

I'm working on a SatCLIP project with learned activation functions at `/Users/hamzaiqbal/grad/learned_activation/satclip`.

## Essential Docs (READ THESE FIRST)
1. `CLAUDE.md` - Quick reference for paths, commands, job workflow
2. `EXPERIMENTS.md` - Experiment tracking and status
3. `experiments/configs/CONFIG_REFERENCE.md` - All config options
4. `env.sh` - Environment variables (source this on HPC)

## Key Architecture
- Paths are managed via `env.sh` (bash) and `experiments/utils/paths.py` (Python)
- Paths auto-detect HPC (`/engrfs` exists) vs local
- Configs use `null` for paths that should be auto-detected

## Current Training Setup (Quick Facts)
- **Baseline SH parity**: use `model.encoding.type: satclip_sh` (or `sh_v2`) for OG SatCLIP SH.
- **SIREN baseline fixed**: proper SIREN weight init is now applied when `activation.type: siren`.
- **Shared activations**: `model.activation.shared: true` (default) shares one activation across layers.
- **HF multispectral default**: contrastive configs expect `dataset: satclip_multispectral` for /10000 + B10 padding.
- **ViT crop size**: `moco_vit16` auto-forces `crop_size=224` in the HF multispectral datamodule.
- **Optional DDP negatives**: `training.gather_negatives: true` enables global negatives.
- **Optional activation freeze**: `training.activation_freeze_epoch: N` freezes activation params after epoch N.

## New Baseline Configs
- `experiments/configs/experiments/contrastive_satclip_baseline.yaml`: SatCLIP baseline (SIREN + SatCLIP SH + MoCo ResNet18).

## HPC Connection
```bash
ssh hiqbal@shell.engr.wustl.edu
Standard Workflow

# 1. Sync code
cd /Users/hamzaiqbal/grad/learned_activation/satclip
git add -A && git commit -m "msg" && git push origin main

# 2. Pull on HPC and run
ssh hiqbal@shell.engr.wustl.edu 'cd /engrfs/project/jacobsn/hiqbal/src/satclip && git pull origin main && sbatch experiments/scripts/slurm/submit_contrastive.sh --activation spline --short'

# Baseline SatCLIP-style run (SIREN + SatCLIP SH)
ssh hiqbal@shell.engr.wustl.edu 'cd /engrfs/project/jacobsn/hiqbal/src/satclip && sbatch experiments/scripts/slurm/submit_contrastive.sh --config experiments/configs/experiments/contrastive_satclip_baseline.yaml'

# 3. Monitor
ssh hiqbal@shell.engr.wustl.edu 'squeue -u hiqbal'
Key Environment Variables (auto-set by env.sh)
SATCLIP_ROOT: /engrfs/project/jacobsn/hiqbal/src/satclip
SATCLIP_DATA_DIR: /engrfs/tmp/jacobsn/hiqbal_satclip
SATCLIP_DATASET_PATH: $SATCLIP_DATA_DIR/satclip_hf_preprocessed
SATCLIP_LOGS_DIR: $SATCLIP_DATA_DIR/logs
SATCLIP_CONDA_ENV: /engrfs/project/jacobsn/hiqbal/conda/envs/satclip
Important: Before making path changes
Read env.sh and experiments/utils/paths.py first
Never hardcode paths in configs - use null for auto-detection
Test with sbatch experiments/scripts/slurm/test_infra.sh before big runs
Current Status
Check EXPERIMENTS.md for current experiment phase and completed runs.
