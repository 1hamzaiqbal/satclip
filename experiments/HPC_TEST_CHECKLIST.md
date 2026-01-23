# HPC Test Checklist for Contrastive Training

Run these tests on the HPC to verify the new contrastive training infrastructure works correctly.

---

## 1. Quick Environment Verification (interactive session)

```bash
# Get interactive session
srun -p condo-jacobsn -A engr-lab-jacobsn --gpus a40:1 -c 8 --mem=32G --pty /bin/bash

# Activate env
source /engrfs/project/jacobsn/hiqbal/conda/envs/satclip/bin/activate

# Verify new dependencies
python -c "import torchgeo; import timm; print('torchgeo + timm OK')"
python -c "from datasets import load_dataset; print('datasets OK')"
```

**If missing packages:**
```bash
pip install torchgeo timm datasets
```

---

## 2. Test Module Imports

```bash
cd /engrfs/project/jacobsn/hiqbal/src/satclip

# Test ContrastiveLearningModule
python -c "
from experiments.models.lightning_module import ContrastiveLearningModule
model = ContrastiveLearningModule(vision_encoder='moco_resnet18', activation_type='spline')
print(f'Location encoder activation: {model.location_encoder.activation_type}')
print('ContrastiveLearningModule OK!')
"
```

---

## 3. Test Regression Training (quick, no data download)

```bash
# Should work immediately with synthetic/elevation data
python -m experiments.train \
    --config experiments/configs/experiments/elevation.yaml \
    --training.max_epochs=2 \
    --data.n_samples=500
```

**Expected**: Completes 2 epochs, prints best val_loss.

---

## 4. Test Contrastive Data Loading (downloads ~50GB first time)

```bash
# Set cache to scratch
export HF_HOME=/scratch/$USER/hf_cache

# Test data loading
python -c "
from experiments.data import SatCLIPHuggingFaceDataModule
dm = SatCLIPHuggingFaceDataModule(batch_size=4, cache_dir='/scratch/$USER/satclip_data')
dm.prepare_data()  # Downloads data
dm.setup()
batch = next(iter(dm.train_dataloader()))
print(f'Image shape: {batch[\"image\"].shape}')
print(f'Point shape: {batch[\"point\"].shape}')
print('Data loading OK!')
"
```

**Expected output:**
```
Image shape: torch.Size([4, 13, 256, 256])  # or similar
Point shape: torch.Size([4, 2])
Data loading OK!
```

---

## 5. Test Contrastive Training (quick test)

```bash
python -m experiments.train \
    --config experiments/configs/experiments/contrastive.yaml \
    --training.max_epochs=2 \
    --data.batch_size=16 \
    --training.accumulate_grad_batches=1 \
    --data.cache_dir=/scratch/$USER/satclip_data
```

**Expected**: Completes 2 epochs, shows train_loss, train_acc_img, val_loss.

---

## 6. SLURM Batch Test

```bash
# Exit interactive session first
exit

# Submit test job
cd /engrfs/project/jacobsn/hiqbal/src/satclip
sbatch experiments/scripts/slurm/submit_contrastive.sh --test

# Check status
squeue -u $USER

# Watch logs (replace JOB_ID)
tail -f logs/satclip_contrastive_*.out
```

---

## 7. Full Contrastive Training (after tests pass)

```bash
# SIREN + SH (original SatCLIP)
sbatch experiments/scripts/slurm/submit_contrastive.sh

# Spline + SH
sbatch experiments/scripts/slurm/submit_contrastive.sh --activation spline

# ReLU + SH (often best based on regression experiments)
sbatch experiments/scripts/slurm/submit_contrastive.sh --activation relu

# Higher resolution SH
sbatch experiments/scripts/slurm/submit_contrastive.sh --encoding sh_l20

# ViT vision encoder (better quality, slower)
sbatch experiments/scripts/slurm/submit_contrastive.sh --vision moco_vit16
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: torchgeo` | `pip install torchgeo timm` |
| `ModuleNotFoundError: datasets` | `pip install datasets` |
| CUDA out of memory | Reduce `--data.batch_size` to 8 or 4 |
| Data download hangs | Set `export HF_TOKEN=your_token` for rate limits |
| `placeholder vision encoder` warning | Install torchgeo and timm |

---

## Files Changed

| File | Purpose |
|------|---------|
| `experiments/models/lightning_module.py` | ContrastiveLearningModule with MoCo vision encoders |
| `experiments/train.py` | Contrastive training support |
| `experiments/configs/experiments/contrastive.yaml` | Full config with docs |
| `experiments/scripts/slurm/submit_contrastive.sh` | HPC training script |
| `experiments/README.md` | Updated documentation |
