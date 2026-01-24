#!/usr/bin/env python
"""Verify checkpoint saving works with the actual training setup.

This test mimics the real training config but with:
- 5 epochs (to verify every_n_epochs=2 works)
- 5 train batches per epoch
- 2 val batches per epoch
- Checkpoints every 2 epochs

Expected checkpoints after test:
- epoch=1-val_loss=X.XXXX.ckpt (epoch 2)
- epoch=3-val_loss=X.XXXX.ckpt (epoch 4)
- last.ckpt
"""

import os
import sys
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from experiments.models.lightning_module import ContrastiveLearningModule
from experiments.data import HFMultispectralDataModule
from experiments.callbacks import EpochLoggerCallback


def test_checkpoints():
    """Test that checkpoints are saved correctly."""
    print("=" * 60)
    print("Checkpoint Saving Verification Test")
    print("=" * 60)

    # Paths
    dataset_path = "/engrfs/tmp/jacobsn/hiqbal_satclip/satclip_hf_preprocessed"
    output_dir = "/engrfs/tmp/jacobsn/hiqbal_satclip/test_checkpoint_save"

    # Clean previous test output
    if os.path.exists(output_dir):
        print(f"Cleaning previous test output: {output_dir}")
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Dataset: {dataset_path}")
    print(f"Output dir: {output_dir}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    # Create data module
    print("\nCreating data module...")
    datamodule = HFMultispectralDataModule(
        dataset_path=dataset_path,
        batch_size=32,
        num_workers=2,
        crop_size=224,
        val_split=0.1,
        preprocessed=True,
        seed=42,
    )
    datamodule.setup()
    print(f"Train samples: {len(datamodule.train_dataset)}")
    print(f"Val samples: {len(datamodule.val_dataset)}")

    # Create model - same as contrastive training
    print("\nCreating model (spline activation)...")
    model = ContrastiveLearningModule(
        vision_encoder="moco_resnet18",
        embed_dim=256,
        freeze_vision=True,
        encoding_type="spherical_harmonics",
        encoding_config={"legendre_polys": 10, "harmonics_calculation": "analytic"},
        hidden_dim=256,
        num_layers=2,
        activation_type="spline",  # Use spline like real training
        activation_config={
            "n_knots": 15,
            "input_range": [-3.0, 3.0],
            "init": "relu",
            "learnable_positions": False,
        },
        learning_rate=1e-4,
        weight_decay=0.01,
        temperature=0.07,
    )
    print("Model created!")

    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name="",
        version="",
    )

    # Checkpoint callback - save every 2 epochs
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_epochs=2,  # Save every 2 epochs
        verbose=True,
    )

    # Epoch logger for clear output
    epoch_logger = EpochLoggerCallback()

    # Trainer
    print("\nCreating trainer...")
    print(f"  Max epochs: 5")
    print(f"  Limit train batches: 5")
    print(f"  Limit val batches: 2")
    print(f"  Checkpoint every: 2 epochs")
    sys.stdout.flush()

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="auto",
        devices=1,
        precision=16,
        callbacks=[checkpoint_callback, epoch_logger],
        logger=logger,
        log_every_n_steps=1,
        enable_progress_bar=True,
        limit_train_batches=5,
        limit_val_batches=2,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training (5 epochs)...")
    print("=" * 60)
    sys.stdout.flush()

    trainer.fit(model, datamodule=datamodule)

    # Verify checkpoints
    print("\n" + "=" * 60)
    print("Verifying checkpoints...")
    print("=" * 60)

    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')])

    print(f"\nCheckpoints found ({len(checkpoints)}):")
    for ckpt in checkpoints:
        ckpt_path = os.path.join(checkpoint_dir, ckpt)
        size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
        print(f"  - {ckpt} ({size_mb:.2f} MB)")

    # Expected checkpoints
    # With every_n_epochs=2, save_top_k=3, save_last=True, 5 epochs:
    # - Epoch 2 (index 1): first checkpoint
    # - Epoch 4 (index 3): second checkpoint
    # - last.ckpt after epoch 5
    expected_min = 2  # At least last.ckpt and one periodic checkpoint

    if len(checkpoints) >= expected_min:
        print("\n✓ CHECKPOINT TEST PASSED!")
        print(f"  Expected at least {expected_min} checkpoints, found {len(checkpoints)}")
        return True
    else:
        print("\n✗ CHECKPOINT TEST FAILED!")
        print(f"  Expected at least {expected_min} checkpoints, found {len(checkpoints)}")
        return False


if __name__ == "__main__":
    pl.seed_everything(42)
    success = test_checkpoints()
    sys.exit(0 if success else 1)
