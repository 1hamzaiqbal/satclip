# Experiment Tracking

This file tracks all experiments, their configurations, and results.

---

## Current Experiment: Learned Activation Functions for Location Encoding

**Goal**: Compare learned spline activations vs fixed activations (ReLU, GELU, SIREN) in SatCLIP-style contrastive learning.

**Hypothesis**: Learnable spline activations can adapt to the specific input distribution of spherical harmonic position encodings, potentially learning smoother or more efficient representations than fixed activations.

---

## Experiment Configurations

### Baseline: SIREN (Original SatCLIP)
```yaml
model:
  activation:
    type: siren
    w0: 1.0
    w0_initial: 30.0
```

### Experiment: Spline Activation
```yaml
model:
  activation:
    type: spline
    n_knots: 15
    input_range: [-3.0, 3.0]
    init: relu
    learnable_positions: false
```

### Control: ReLU
```yaml
model:
  activation:
    type: relu
```

---

## Completed Runs

| Job ID | Date | Config | Epochs | Status | Notes |
|--------|------|--------|--------|--------|-------|
| 18475 | 2026-01-24 | test_infra | 2 | PASSED | Infrastructure test |
| 18479 | 2026-01-24 | checkpoint_test | 5 | PASSED | Checkpoint saving verified |
| 18490 | 2026-01-24 | test_infra | 2 | PASSED | Env var workflow verified |
| 18498 | 2026-01-24 | spline + short | 50 | COMPLETED | Pre-fix exploratory (see below) |

### Job 18498: Spline Short Run (Pre-Baseline-Fix)

**Config**: `contrastive_multispectral.yaml` + `spline.yaml` + `contrastive_short.yaml`

**Results**:
- **Val Loss**: 5.66 → 2.47 (best: epoch 39, val_loss=2.4714)
- **Training Time**: ~2.5 min/epoch, ~2h 15min total
- **Checkpoint**: `/engrfs/tmp/jacobsn/hiqbal_satclip/logs/contrastive_multispectral/version_*/checkpoints/epoch=39-val_loss=2.4714.ckpt`

**Hardware**: A40 GPU, 192GB RAM, 16 CPUs, batch=256, accum=4 (effective=1024)

**Caveats** (treat as exploratory, not comparable to corrected baselines):
1. Encoding may have been `spherical_harmonics` (simplified) NOT `satclip_sh` (original parity)
2. Activation sharing status unknown - may not have used shared activation
3. Convergence behavior is informative, absolute metrics need re-validation

---

## Planned Experiments

### Phase 1: Infrastructure Validation (COMPLETE)
- [x] Data loading from HF Arrow dataset
- [x] Checkpoint saving to explicit directory
- [x] TensorBoard logging
- [x] Epoch progress logging
- [x] Environment variable path management

---

## Phase 2: Corrected Baseline vs Spline Comparison

### Experimental Design

**Objective**: Fair comparison of SIREN (SatCLIP default) vs learned spline activations.

**Controlled Variables** (same for all runs):
| Parameter | Value |
|-----------|-------|
| Encoding | `satclip_sh` (L=10, 100 dims, original SatCLIP parity) |
| Vision Encoder | `moco_resnet50` (frozen, 13-channel multispectral) |
| Hidden/Output Dim | 512 |
| Network Layers | 2 |
| Temperature | 0.07 |
| Batch Size | 256 (effective 1024 with accum=4) |
| Learning Rate | 0.0001 |
| Weight Decay | 0.01 |
| Scheduler | warmup_cosine (10 warmup epochs) |
| Dataset | satclip_multispectral (HF preprocessed) |

**Independent Variable**: Activation function
- **SIREN** (baseline): w0=1.0, w0_initial=30.0, proper weight init
- **Spline** (experiment): k=15, range=[-3,3], init=relu, shared=true

### Experiment 2.1: SIREN Baseline (Short)
```bash
# Command
sbatch experiments/scripts/slurm/submit_contrastive.sh --activation siren --short

# Configs (layered)
- experiments/configs/experiments/contrastive_multispectral.yaml (base)
- experiments/configs/activations/siren.yaml (activation)
- experiments/configs/experiments/contrastive_short.yaml (50 epochs)
```
**Expected**: Establishes baseline val_loss for 50-epoch comparison.

### Experiment 2.2: Spline Activation (Short)
```bash
# Command
sbatch experiments/scripts/slurm/submit_contrastive.sh --activation spline --short

# Configs (layered)
- experiments/configs/experiments/contrastive_multispectral.yaml (base)
- experiments/configs/activations/spline.yaml (activation)
- experiments/configs/experiments/contrastive_short.yaml (50 epochs)
```
**Expected**: Compare convergence speed and final val_loss to SIREN baseline.

### Success Criteria
- Both runs complete without OOM/errors
- TensorBoard logs available for comparison
- Spline checkpoints available for shape visualization
- Clear winner or comparable performance with interpretable learned shapes

---

### Phase 3: Full Training (500 epochs)
- [ ] SIREN baseline (500 epochs) - if Phase 2 looks promising
- [ ] Spline (500 epochs) - if Phase 2 looks promising
- [ ] Spline shape evolution analysis (checkpoints every 10 epochs)
- [ ] Globe embedding visualization

### Phase 4: Ablations (after Phase 3)
- [ ] Spline knot count: k=10, 15, 20, 30
- [ ] Input range: [-2,2], [-3,3], [-4,4]
- [ ] Learnable knot positions
- [ ] ReLU/GELU controls (sanity check)

---

## Key Metrics to Track

1. **val_loss**: Primary metric (InfoNCE contrastive loss)
2. **val_acc_img**: Image-to-location retrieval accuracy
3. **val_acc_loc**: Location-to-image retrieval accuracy
4. **Spline shapes**: How do learned activations evolve?
5. **Embedding structure**: Do embeddings show geographic clustering?

---

## Analysis Workflow

After training completes:

1. **Extract checkpoints**: Copy from HPC to local
   ```bash
   scp -r hiqbal@shell.engr.wustl.edu:/engrfs/tmp/jacobsn/hiqbal_satclip/logs/contrastive_multispectral/<RUN>/checkpoints/ ./checkpoints/
   ```

2. **Visualize spline evolution**:
   ```bash
   python -m experiments.scripts.analysis.visualize_epochs \
       --checkpoint_dir ./checkpoints \
       --output_dir ./epoch_visualizations
   ```

3. **Create GIF**:
   ```bash
   python -m experiments.scripts.analysis.create_evolution_gif \
       --input_dir ./epoch_visualizations \
       --output ./training_evolution.gif
   ```

4. **Compare activations**: (TODO: create comparison script)

---

## Notes

### 2026-01-24
- Infrastructure verified working after fixing checkpoint saving
- Added EpochLoggerCallback for clear progress in SLURM logs
- Added modular path management with env.sh and paths.py
- Test jobs 18475, 18479, 18490 all passed

### 2026-01-24 (Caveats for runs started before baseline fixes)
- SH encoding likely used simplified `spherical_harmonics` instead of SatCLIP parity (`satclip_sh`).
- SIREN baseline did not use proper SIREN weight initialization.
- Contrastive runs using `dataset: satclip` lacked /10000 normalization and optional B10 padding (MoCo mismatch).
- Activations were not shared across layers by default (paper baseline expects shared).
- ViT runs may have used non-224 crops unless explicitly set.
- DDP runs used local negatives only (no global all_gather).
- These runs are not directly comparable to the corrected baselines; treat as exploratory.

---

## References

- Original SatCLIP paper: [Klemmer et al., 2023]
- TTE (Time-To-Event) repo: Pattern reference for PyTorch Lightning
- B-spline theory: [de Boor, 1978]
