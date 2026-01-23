# HPC Instructions for SatCLIP Preprocessing and Testing

## Overview

This document provides instructions for:
1. Extracting and preprocessing the S2K multispectral Sentinel-2 dataset
2. Running tests to verify the setup
3. Training models with the preprocessed data

## Data Paths (Update These!)

**Current default paths in scripts:**
```
DATA_DIR=/projects/bdbk/cherd/data/satclip_manual
```

**Expected directory structure:**
```
$DATA_DIR/
├── archives/           # Original tar.xz files (s2k_all_*.tar.xz)
├── raw_tifs/           # Extracted TIF files (after build script)
├── satclip_hf/         # HuggingFace Arrow dataset (after build script)
└── satclip_hf_preprocessed/  # Normalized + padded dataset (after preprocess script)
```

**To update paths:** Edit the following files:
- `experiments/scripts/slurm/preprocess_s2k.sh` - lines 48-52
- `experiments/configs/experiments/contrastive_multispectral.yaml` - line 72

## Step 1: Start Preprocessing Pipeline

The preprocessing has 3 stages:
1. **Extract**: Unpack tar.xz archives to TIF files
2. **Build**: Convert TIF files to HuggingFace Arrow format
3. **Preprocess**: Normalize values and pad to 13 channels

### Submit the job:

```bash
cd /path/to/learned_activation/satclip

# Full pipeline (all 3 stages)
sbatch experiments/scripts/slurm/preprocess_s2k.sh

# Skip extraction if TIFs already extracted
sbatch experiments/scripts/slurm/preprocess_s2k.sh --skip-extract

# Skip both extraction and build (only preprocess)
sbatch experiments/scripts/slurm/preprocess_s2k.sh --skip-extract --skip-build
```

**Estimated time:** 4-8 hours for full pipeline (depends on I/O)

### Monitor progress:

```bash
# Check job status
squeue -u $USER

# View output logs
tail -f logs/preprocess_s2k_*.out

# View error logs
tail -f logs/preprocess_s2k_*.err
```

## Step 2: Run HPC Tests

After preprocessing completes, run the test suite to verify everything works:

```bash
cd /path/to/learned_activation/satclip

# Activate conda environment
conda activate satclip

# Run test suite
python experiments/scripts/local/test_setup.py
```

### What the tests verify:
- All module imports work
- Activation functions (relu, spline) work correctly
- Positional encodings (SH, raw, cartesian3d) work correctly
- Location encoder builds and runs
- Coordinate utilities (latlon_to_sphere, etc.) work correctly
- RANGE evaluation framework imports and runs
- Lightning module instantiates correctly
- Full training loop completes (1 epoch, tiny dataset)

### Additional HPC-specific tests:

```bash
# Test that preprocessed dataset can be loaded
python -c "
from experiments.data import HFMultispectralDataModule
dm = HFMultispectralDataModule(
    dataset_path='/projects/bdbk/cherd/data/satclip_manual/satclip_hf_preprocessed',
    batch_size=32,
    num_workers=4,
    preprocessed=True,
    pad_to_13_channels=False,  # Already padded
)
dm.setup()
batch = next(iter(dm.train_dataloader()))
print(f'Image batch shape: {batch[\"image\"].shape}')
print(f'Coords batch shape: {batch[\"coords\"].shape}')
print(f'Image value range: [{batch[\"image\"].min():.3f}, {batch[\"image\"].max():.3f}]')
"

# Test GPU availability
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"

# Test vision encoder loading
python -c "
from experiments.models.lightning_module import ContrastiveLearningModule
model = ContrastiveLearningModule(
    vision_encoder='moco_resnet50',
    embed_dim=512,
)
print(f'Model loaded successfully')
print(f'Vision encoder output dim: {model.embed_dim}')
"
```

## Step 3: Training (After Preprocessing Complete)

### Quick test run (fast_dev_run):

```bash
python -m experiments.train \
    --config experiments/configs/experiments/contrastive_multispectral.yaml \
    --trainer.fast_dev_run=true
```

### Full training:

```bash
# Interactive (for testing)
python -m experiments.train \
    --config experiments/configs/experiments/contrastive_multispectral.yaml

# Or submit as SLURM job
sbatch experiments/scripts/slurm/submit_contrastive.sh
```

## Step 4: Evaluation

After training completes:

```bash
sbatch experiments/scripts/slurm/eval_range.sh \
    --model_path /path/to/checkpoint.pt

# Or for specific tasks only:
sbatch experiments/scripts/slurm/eval_range.sh \
    --model_path /path/to/checkpoint.pt \
    --tasks biome,elevation,checker_100
```

## Troubleshooting

### Common issues:

1. **"FileNotFoundError: No such file or directory"**
   - Check that DATA_DIR path is correct
   - Verify preprocessing completed successfully

2. **"CUDA out of memory"**
   - Reduce batch_size in config
   - Reduce accumulate_grad_batches

3. **"ImportError: No module named 'experiments'"**
   - Make sure you're in the satclip root directory
   - Or add to PYTHONPATH: `export PYTHONPATH=$PYTHONPATH:/path/to/satclip`

4. **Preprocessing stalls**
   - Check disk space: `df -h`
   - Check I/O with: `iotop`

### Getting help:

- Check logs in `logs/` directory
- Run test_setup.py to identify specific failures
