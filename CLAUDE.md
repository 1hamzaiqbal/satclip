# SatCLIP Project - Quick Reference

## VERIFIED: Infrastructure Working

Last verified: 2026-01-24 16:32 CST (Job 18479)
- **Checkpoints**: WORKING (saves to explicit `checkpoints/` subdirectory)
- **TensorBoard**: WORKING
- **Data loading**: WORKING (HF dataset loads correctly)
- **EpochLogger**: WORKING (prints `Epoch X/Y | Val Loss: Z.ZZZZ | Time: Xm Ys`)

### Current Training Setup (2026-01-24 updates)
- **SatCLIP SH parity**: use `model.encoding.type: satclip_sh` (or `sh_v2`) for OG SatCLIP positional encoding.
- **SIREN baseline fixed**: proper SIREN weight init is now applied when `activation.type: siren`.
- **Shared activations**: `model.activation.shared: true` shares one activation across layers (paper default).
- **HF multispectral default**: contrastive configs default to `dataset: satclip_multispectral` for /10000 normalization + B10 padding.
- **ViT crop size**: `moco_vit16` auto-forces `crop_size=224` in HF multispectral datamodule.
- **Optional DDP negatives**: `training.gather_negatives: true` enables global negatives across ranks.
- **Optional activation freeze**: `training.activation_freeze_epoch: N` freezes activation params after epoch N.

### Checkpoint Test Results (Job 18479)
```
5 epochs with 5 train batches each, checkpoints every 2 epochs:
- epoch=1-val_loss=3.4388.ckpt (46.15 MB)
- epoch=3-val_loss=3.3412.ckpt (46.16 MB)
- last.ckpt (46.16 MB)
```

### Estimated Training Speed
- With full dataset (84K samples), batch 512, 4x accumulation: ~41 steps/epoch
- Each epoch: ~2-3 minutes on A40 with mixed precision
- Epoch 10 checkpoint appears around 25-30 min into training

---

## Current Active Job

_No active job_

### Quick Check Commands
```bash
# Check status
ssh hiqbal@shell.engr.wustl.edu 'squeue -u hiqbal'

# View latest job output (replace JOBID)
ssh hiqbal@shell.engr.wustl.edu 'cat /engrfs/project/jacobsn/hiqbal/src/satclip/logs/satclip_contrastive_<JOBID>.out'
```

---

## Environment Variables

All paths are managed via `env.sh` and `experiments/utils/paths.py`.
Scripts auto-detect HPC vs local environment.

### Setup
```bash
# Source env.sh (auto-detects environment)
source env.sh

# Or force specific environment
source env.sh hpc    # HPC paths
source env.sh local  # Local paths

# Check what's set
echo $SATCLIP_ROOT
echo $SATCLIP_DATA_DIR
```

### Key Variables
| Variable | HPC Value | Description |
|----------|-----------|-------------|
| `SATCLIP_ROOT` | `/engrfs/project/jacobsn/hiqbal/src/satclip` | Project root |
| `SATCLIP_DATA_DIR` | `/engrfs/tmp/jacobsn/hiqbal_satclip` | Data/logs base |
| `SATCLIP_DATASET_PATH` | `$SATCLIP_DATA_DIR/satclip_hf_preprocessed` | HF dataset |
| `SATCLIP_LOGS_DIR` | `$SATCLIP_DATA_DIR/logs` | Training logs |
| `SATCLIP_CONDA_ENV` | `/engrfs/project/jacobsn/hiqbal/conda/envs/satclip` | Conda env |

### In Python
```python
from experiments.utils.paths import get_paths, is_hpc
paths = get_paths()
print(paths.dataset)  # Auto-detected dataset path
print(paths.logs)     # Auto-detected logs path
```

---

## Key Paths (Reference)

### Repository
| Description | Path |
|-------------|------|
| SatCLIP repo | `$SATCLIP_ROOT` |
| Local mirror | `/Users/hamzaiqbal/grad/learned_activation/satclip` |
| Conda env | `$SATCLIP_CONDA_ENV` |

### Data & Outputs (Temp Pool - Unlimited Quota)
| Description | Path |
|-------------|------|
| Preprocessed HF dataset | `/engrfs/tmp/jacobsn/hiqbal_satclip/satclip_hf_preprocessed` |
| Training logs/checkpoints | `/engrfs/tmp/jacobsn/hiqbal_satclip/logs` |
| Test outputs | `/engrfs/tmp/jacobsn/hiqbal_satclip/test_output` |
| HF cache | `/engrfs/tmp/jacobsn/hiqbal_satclip/hf_cache` |

### Project Folder (Limited Quota - Use for Code Only)
| Description | Path |
|-------------|------|
| SLURM logs | `/engrfs/project/jacobsn/hiqbal/src/satclip/logs/` |

---

## Quick Commands

### SSH Access
```bash
ssh hiqbal@shell.engr.wustl.edu
```

### Check Job Status
```bash
# Quick status
ssh hiqbal@shell.engr.wustl.edu 'squeue -u hiqbal'

# Detailed job info
ssh hiqbal@shell.engr.wustl.edu 'sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS,MaxVMSize'
```

### View Logs
```bash
# SLURM job output (replace JOBID)
ssh hiqbal@shell.engr.wustl.edu 'cat /engrfs/project/jacobsn/hiqbal/src/satclip/logs/satclip_contrastive_<JOBID>.out'

# SLURM job errors
ssh hiqbal@shell.engr.wustl.edu 'cat /engrfs/project/jacobsn/hiqbal/src/satclip/logs/satclip_contrastive_<JOBID>.err'

# Recent log files
ssh hiqbal@shell.engr.wustl.edu 'ls -lt /engrfs/project/jacobsn/hiqbal/src/satclip/logs/ | head -20'
```

### Check Training Progress
```bash
# Find latest run directory
ssh hiqbal@shell.engr.wustl.edu 'ls -lt /engrfs/tmp/jacobsn/hiqbal_satclip/logs/contrastive_multispectral/ | head -5'

# Check TB file (growing = training active)
ssh hiqbal@shell.engr.wustl.edu 'ls -lh /engrfs/tmp/jacobsn/hiqbal_satclip/logs/contrastive_multispectral/*/events.out.tfevents.*'

# Check for checkpoints
ssh hiqbal@shell.engr.wustl.edu 'ls -la /engrfs/tmp/jacobsn/hiqbal_satclip/logs/contrastive_multispectral/*/checkpoints/'
```

### Find Checkpoints
```bash
ssh hiqbal@shell.engr.wustl.edu 'find /engrfs/tmp/jacobsn/hiqbal_satclip -name "*.ckpt" 2>/dev/null'
```

### Check Disk Quota
```bash
ssh hiqbal@shell.engr.wustl.edu 'df -h /engrfs/project/jacobsn/hiqbal'
```

---

## Job Execution Workflow

### Step 1: Sync Code to HPC
```bash
# From local machine - commit and push
cd /Users/hamzaiqbal/grad/learned_activation/satclip
git add -A && git commit -m "message" && git push origin main

# Pull on HPC
ssh hiqbal@shell.engr.wustl.edu 'cd /engrfs/project/jacobsn/hiqbal/src/satclip && git pull origin main'
```

### Step 2: Submit Job
All jobs are submitted from `$SATCLIP_ROOT` directory.

```bash
# Baseline SatCLIP-style run (SIREN + SatCLIP SH)
ssh hiqbal@shell.engr.wustl.edu 'cd /engrfs/project/jacobsn/hiqbal/src/satclip && sbatch experiments/scripts/slurm/submit_contrastive.sh --config experiments/configs/experiments/contrastive_satclip_baseline.yaml'

# Quick test (verifies infrastructure, ~2 min)
ssh hiqbal@shell.engr.wustl.edu 'cd /engrfs/project/jacobsn/hiqbal/src/satclip && sbatch experiments/scripts/slurm/test_infra.sh'

# Checkpoint test (verifies checkpoints save, ~3 min)
ssh hiqbal@shell.engr.wustl.edu 'cd /engrfs/project/jacobsn/hiqbal/src/satclip && sbatch experiments/scripts/slurm/test_checkpoint.sh'

# Short training run (50 epochs, ~2 hours)
ssh hiqbal@shell.engr.wustl.edu 'cd /engrfs/project/jacobsn/hiqbal/src/satclip && sbatch --mem=128G experiments/scripts/slurm/submit_contrastive.sh --activation spline --short'

# Full training run (500 epochs, ~20 hours)
ssh hiqbal@shell.engr.wustl.edu 'cd /engrfs/project/jacobsn/hiqbal/src/satclip && sbatch --mem=128G experiments/scripts/slurm/submit_contrastive.sh --activation spline'
```

### Step 3: Monitor Job
```bash
# Check if running
ssh hiqbal@shell.engr.wustl.edu 'squeue -u hiqbal'

# View output (replace JOBID)
ssh hiqbal@shell.engr.wustl.edu 'tail -50 /engrfs/project/jacobsn/hiqbal/src/satclip/logs/satclip_contrastive_<JOBID>.out'

# Check for checkpoints
ssh hiqbal@shell.engr.wustl.edu 'ls -la /engrfs/tmp/jacobsn/hiqbal_satclip/logs/contrastive_multispectral/*/checkpoints/'
```

### submit_contrastive.sh Options
| Option | Description |
|--------|-------------|
| `--activation <type>` | relu, gelu, siren, spline |
| `--encoding <type>` | sh_l10, sh_l20 |
| `--vision <type>` | moco_resnet18, moco_resnet50, moco_vit16 |
| `--short` | 50 epochs (monitoring run) |
| `--test` | 2 epochs (quick validation) |
| `--data.num_workers=N` | Override num_workers |

---

## Config Files

### Main Configs
- `experiments/configs/experiments/contrastive_multispectral.yaml` - Full 500-epoch config
- `experiments/configs/experiments/contrastive_short.yaml` - 50-epoch monitoring config

### Activation Configs
- `experiments/configs/activations/relu.yaml`
- `experiments/configs/activations/gelu.yaml`
- `experiments/configs/activations/siren.yaml`
- `experiments/configs/activations/spline.yaml`

### Spline Configuration (in spline.yaml)
```yaml
model:
  activation:
    type: "spline"
    n_knots: 15
    input_range: [-3.0, 3.0]
    init: "relu"
    learnable_positions: false
```

---

## Post-Training Analysis

### Generate Epoch Visualizations
```bash
python -m experiments.scripts.analysis.visualize_epochs \
    --checkpoint_dir /path/to/checkpoints \
    --output_dir epoch_visualizations
```

### Create Evolution GIF
```bash
python -m experiments.scripts.analysis.create_evolution_gif \
    --input_dir epoch_visualizations \
    --output training_evolution.gif
```

---

## Checkpoint Schedule

- Short run (50 epochs): Save every 10 epochs, save_top_k=3
- Full run (500 epochs): Save every 10 epochs, save_top_k=3

### Expected Checkpoint Files
```
checkpoints/
  epoch=09-val_loss=X.XXXX.ckpt
  epoch=19-val_loss=X.XXXX.ckpt
  ...
  last.ckpt
```

---

## Troubleshooting

### OOM Errors
- Increase memory: `sbatch --mem=128G` or `--mem=192G`
- Reduce batch size in config
- Reduce `num_workers`

### Quota Issues
- Use temp pool for large outputs: `/engrfs/tmp/jacobsn/hiqbal_satclip/`
- Check quota: `df -h /engrfs/project/jacobsn/hiqbal`

### No Checkpoints Appearing
1. Check job is still running: `squeue -u hiqbal`
2. Check for errors: View SLURM `.err` file
3. Verify checkpoint directory in verbose output
4. Checkpoints only save every N epochs (default: 10)

### Device Mismatch Errors
- Ensure tensors are moved to correct device before operations
- Use `next(module.parameters()).device` for module's actual device

---

## Sync Commands (Local ↔ HPC)

### Git Sync (Preferred)
```bash
# Local: commit and push
cd /Users/hamzaiqbal/grad/learned_activation/satclip
git add -A && git commit -m "message" && git push origin main

# HPC: pull
ssh hiqbal@shell.engr.wustl.edu 'cd /engrfs/project/jacobsn/hiqbal/src/satclip && git pull origin main'
```

### Quick File Sync (when needed)
```bash
# Single file
scp /Users/hamzaiqbal/grad/learned_activation/satclip/experiments/train.py hiqbal@shell.engr.wustl.edu:/engrfs/project/jacobsn/hiqbal/src/satclip/experiments/

# Directory
rsync -avz --exclude='.git' --exclude='__pycache__' /Users/hamzaiqbal/grad/learned_activation/satclip/experiments/ hiqbal@shell.engr.wustl.edu:/engrfs/project/jacobsn/hiqbal/src/satclip/experiments/
```

---

## Reference: TTE Patterns to Follow

Based on TTE repo analysis (see local `/Users/hamzaiqbal/grad/learned_activation/TTE`):

1. **PyTorch Lightning**: Clean multi-GPU support, automatic device handling
2. **Hierarchical YAML configs**: Easy experiment variations
3. **Custom callbacks**: Save only improving checkpoints
4. **Modular loss functions**: Easy to enable/disable components
5. **Parameter groups**: Different LRs for different components
6. **Per-component logging**: Detailed loss tracking

---

## Last Updated
2026-01-24 16:35 CST
