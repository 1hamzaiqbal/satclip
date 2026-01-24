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

---

## Planned Experiments

### Phase 1: Infrastructure Validation (COMPLETE)
- [x] Data loading from HF Arrow dataset
- [x] Checkpoint saving to explicit directory
- [x] TensorBoard logging
- [x] Epoch progress logging
- [x] Environment variable path management

### Phase 2: Activation Function Comparison
- [ ] SIREN baseline (50 epochs, monitoring)
- [ ] Spline k=15 (50 epochs, monitoring)
- [ ] ReLU control (50 epochs, monitoring)
- [ ] GELU control (50 epochs, monitoring)

### Phase 3: Full Training
- [ ] Best activation from Phase 2 (500 epochs)
- [ ] Spline shape evolution analysis
- [ ] Globe embedding visualization

### Phase 4: Ablations
- [ ] Spline knot count: k=10, 15, 20, 30
- [ ] Input range: [-2,2], [-3,3], [-4,4]
- [ ] Learnable knot positions

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

---

## References

- Original SatCLIP paper: [Klemmer et al., 2023]
- TTE (Time-To-Event) repo: Pattern reference for PyTorch Lightning
- B-spline theory: [de Boor, 1978]
