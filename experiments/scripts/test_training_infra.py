#!/usr/bin/env python
"""Quick infrastructure test script.

This script runs a minimal training loop to verify:
1. Data loading works
2. Model creation works
3. Checkpointing works
4. Logging works

Usage:
    python -m experiments.scripts.test_training_infra

Expected output:
    - 2 epochs of training
    - Checkpoint files in output directory
    - TensorBoard logs
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from experiments.models.lightning_module import ContrastiveLearningModule
from experiments.data import HFMultispectralDataModule
from experiments.utils.paths import get_paths


def test_training():
    """Run minimal training test."""
    print("=" * 60)
    print("SatCLIP Training Infrastructure Test")
    print("=" * 60)

    # Get paths from environment
    paths = get_paths()
    dataset_path = str(paths.dataset)
    output_dir = str(paths.data_dir / "test_output")

    print(f"\nDataset path: {dataset_path}")
    print(f"Output directory: {output_dir}")

    # Check dataset exists
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        sys.exit(1)
    print("Dataset found!")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Create datamodule with small subset
    print("\nCreating datamodule...")
    datamodule = HFMultispectralDataModule(
        dataset_path=dataset_path,
        batch_size=32,  # Small batch for quick test
        num_workers=2,
        crop_size=224,
        val_split=0.1,
        preprocessed=True,
        seed=42,
    )
    datamodule.setup()
    print(f"  Train samples: {len(datamodule.train_dataset)}")
    print(f"  Val samples: {len(datamodule.val_dataset)}")

    # Create model
    print("\nCreating model...")
    model = ContrastiveLearningModule(
        vision_encoder="moco_resnet18",
        embed_dim=256,
        freeze_vision=True,
        encoding_type="spherical_harmonics",
        encoding_config={"legendre_polys": 10, "harmonics_calculation": "analytic"},
        hidden_dim=256,
        num_layers=2,
        activation_type="relu",  # Simple ReLU for test
        activation_config={},
        learning_rate=1e-4,
        weight_decay=0.01,
        temperature=0.07,
    )
    print("  Model created successfully!")

    # Create logger
    print("\nCreating logger...")
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name="test_run",
        version="test",
    )
    print(f"  Log dir: {logger.log_dir}")

    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="test-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,  # Save every epoch for test
    )

    # Create trainer
    print("\nCreating trainer...")
    trainer = pl.Trainer(
        max_epochs=2,  # Just 2 epochs for test
        accelerator="auto",
        devices=1,
        precision=16,
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        limit_train_batches=10,  # Only 10 batches per epoch for speed
        limit_val_batches=5,
    )
    print("  Trainer created successfully!")

    # Train
    print("\n" + "=" * 60)
    print("Starting test training (2 epochs, 10 batches each)...")
    print("=" * 60)
    sys.stdout.flush()

    trainer.fit(model, datamodule=datamodule)

    # Verify outputs
    print("\n" + "=" * 60)
    print("Verifying outputs...")
    print("=" * 60)

    # Check checkpoints
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    print(f"\nCheckpoints found ({len(checkpoints)}):")
    for ckpt in sorted(checkpoints):
        ckpt_path = os.path.join(checkpoint_dir, ckpt)
        size = os.path.getsize(ckpt_path) / (1024 * 1024)  # MB
        print(f"  - {ckpt} ({size:.2f} MB)")

    # Check TensorBoard logs
    tb_files = list(Path(logger.log_dir).glob("events.out.tfevents.*"))
    print(f"\nTensorBoard files ({len(tb_files)}):")
    for tb_file in tb_files:
        size = tb_file.stat().st_size / 1024  # KB
        print(f"  - {tb_file.name} ({size:.1f} KB)")

    # Final status
    success = len(checkpoints) > 0 and len(tb_files) > 0
    print("\n" + "=" * 60)
    if success:
        print("TEST PASSED!")
        print("All infrastructure components working correctly.")
    else:
        print("TEST FAILED!")
        if len(checkpoints) == 0:
            print("  - No checkpoints saved")
        if len(tb_files) == 0:
            print("  - No TensorBoard logs")
    print("=" * 60)

    return success


if __name__ == "__main__":
    # Set seed
    pl.seed_everything(42)

    # Run test
    success = test_training()
    sys.exit(0 if success else 1)
