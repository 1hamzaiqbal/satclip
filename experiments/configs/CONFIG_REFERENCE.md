# Configuration Reference

This document describes all configuration options for the SatCLIP experiments.

## Config Composition

Configs are loaded in order. Later configs override earlier ones:

```bash
python -m experiments.train \
    --config experiments/configs/base.yaml \                      # Base defaults
    --config experiments/configs/experiments/contrastive_multispectral.yaml \  # Experiment
    --config experiments/configs/activations/spline.yaml \        # Activation
    --config experiments/configs/experiments/contrastive_short.yaml  # Short run
```

## CLI Overrides

Any value can be overridden via CLI:
```bash
python -m experiments.train --config ... \
    --training.max_epochs=100 \
    --model.activation.n_knots=20 \
    --data.batch_size=256
```

---

## experiment

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| name | string | "experiment" | Experiment name (used in logging) |
| seed | int | 42 | Random seed |

---

## model

### model.encoding (Position Encoding)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| type | string | "spherical_harmonics" | Encoding type |
| legendre_polys | int | 10 | L value for SH (gives 2L+1 coeffs) |
| harmonics_calculation | string | "analytic" | "analytic" or "recursive" |

### model.network (MLP)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| hidden_dim | int | 512 | Hidden layer dimension |
| output_dim | int | 512 | Output dimension (match embed_dim) |
| num_layers | int | 2 | Number of hidden layers |

### model.activation

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| type | string | "siren" | Activation function |

**For type="relu"**: No additional options.

**For type="gelu"**: No additional options.

**For type="siren"**:
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| w0 | float | 1.0 | Frequency for hidden layers |
| w0_initial | float | 30.0 | Frequency for first layer |

**For type="spline"**:
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| n_knots | int | 15 | Number of B-spline knots |
| input_range | [float, float] | [-3.0, 3.0] | Input range for knots |
| init | string | "relu" | Initialization: "relu", "gelu", "identity" |
| learnable_positions | bool | false | Whether knot positions are learnable |

### model (Vision Encoder)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| vision_encoder | string | "moco_resnet50" | Vision backbone |
| embed_dim | int | 512 | Embedding dimension |
| freeze_vision | bool | true | Freeze vision encoder |
| temperature | float | 0.07 | Contrastive temperature |

Vision encoder options:
- `moco_resnet18`: MoCo ResNet-18 (smaller, faster)
- `moco_resnet50`: MoCo ResNet-50 (torchgeo)
- `moco_vit16`: MoCo ViT-Small-16 (torchgeo)
- `mae_vit_large16`: MAE ViT-L/16 (SSL4EO-S12)

---

## data

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| task | string | "contrastive" | Task type |
| dataset | string | "satclip_multispectral" | Dataset name |
| use_hf_dataset | bool | true | Use HF Arrow format |
| hf_dataset_path | string | null | Path to HF dataset (auto-detected) |
| crop_size | int | 224 | Image crop size |
| pad_to_13_channels | bool | false | Pad to 13 channels |
| preprocessed | bool | true | Data is pre-normalized |
| batch_size | int | 512 | Batch size |
| num_workers | int | 8 | DataLoader workers |
| val_split | float | 0.1 | Validation split |

---

## training

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| max_epochs | int | 500 | Maximum epochs |
| learning_rate | float | 0.0001 | Learning rate |
| weight_decay | float | 0.01 | Weight decay |
| accumulate_grad_batches | int | 4 | Gradient accumulation |
| gradient_clip_val | float | 1.0 | Gradient clipping |
| scheduler | string | "warmup_cosine" | LR scheduler |
| warmup_epochs | int | 10 | Warmup epochs |
| min_lr | float | 1e-6 | Minimum LR |

### training.early_stopping

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| enabled | bool | false | Enable early stopping |
| patience | int | 50 | Patience (epochs) |
| monitor | string | "val_loss" | Metric to monitor |

---

## logging

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| logger | string | "tensorboard" | Logger type |
| save_dir | string | null | Log directory (auto-detected) |
| name | string | "experiments" | Experiment group name |
| log_every_n_steps | int | 50 | Logging frequency |

### logging.checkpoint

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| save_top_k | int | 3 | Keep top K checkpoints |
| monitor | string | "val_loss" | Metric to monitor |
| mode | string | "min" | "min" or "max" |
| save_last | bool | true | Save last checkpoint |
| every_n_epochs | int | 10 | Save every N epochs |

### logging.spline_viz

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| log_every_n_epochs | int | 10 | Visualization frequency |
| log_on_train_start | bool | true | Log initial shapes |
| input_range | [float, float] | [-4.0, 4.0] | Plot range |

### logging.globe_viz

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| enabled | bool | true | Enable globe viz |
| log_every_n_epochs | int | 25 | Visualization frequency |
| log_on_train_start | bool | true | Log initial state |
| lat_resolution | int | 90 | Latitude grid points |
| lon_resolution | int | 180 | Longitude grid points |

---

## hardware

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| accelerator | string | "auto" | Device type |
| devices | int | 1 | Number of devices |
| precision | int/string | 16 | Training precision |
