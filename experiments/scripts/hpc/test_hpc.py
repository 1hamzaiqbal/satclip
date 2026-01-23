#!/usr/bin/env python
"""HPC-specific test script for SatCLIP preprocessing and training pipeline.

This script tests:
1. All module imports
2. GPU availability
3. Preprocessed dataset loading (if available)
4. Vision encoder loading
5. Full model instantiation
6. Mini training loop

Usage:
    python experiments/scripts/hpc/test_hpc.py
    python experiments/scripts/hpc/test_hpc.py --data-dir /path/to/data
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test all required imports."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)

    modules = [
        ("torch", "PyTorch"),
        ("lightning.pytorch", "PyTorch Lightning"),
        ("datasets", "HuggingFace Datasets"),
        ("torchgeo", "TorchGeo"),
        ("experiments.models.activations", "Activations"),
        ("experiments.models.encodings", "Encodings"),
        ("experiments.models.location_encoder", "Location Encoder"),
        ("experiments.models.lightning_module", "Lightning Module"),
        ("experiments.data", "Data Module"),
        ("experiments.eval_range", "RANGE Evaluation"),
    ]

    all_passed = True
    for module, name in modules:
        try:
            __import__(module)
            print(f"  {name}: OK")
        except ImportError as e:
            print(f"  {name}: FAILED - {e}")
            all_passed = False

    return all_passed


def test_gpu():
    """Test GPU availability."""
    print("\n" + "=" * 60)
    print("Testing GPU...")
    print("=" * 60)

    import torch

    if torch.cuda.is_available():
        print(f"  CUDA available: True")
        print(f"  Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"    Memory: {mem:.1f} GB")
        return True
    else:
        print("  CUDA available: False")
        print("  WARNING: No GPU detected. Training will be slow.")
        return True  # Not a failure, just a warning


def test_preprocessed_dataset(data_dir: str):
    """Test loading the preprocessed HuggingFace dataset."""
    print("\n" + "=" * 60)
    print("Testing preprocessed dataset...")
    print("=" * 60)

    preprocessed_path = Path(data_dir) / "satclip_hf_preprocessed"

    if not preprocessed_path.exists():
        print(f"  Dataset not found at: {preprocessed_path}")
        print("  Skipping dataset tests (preprocessing not complete)")
        return None  # None means skipped, not failed

    try:
        from experiments.data import HFMultispectralDataModule

        dm = HFMultispectralDataModule(
            dataset_path=str(preprocessed_path),
            batch_size=16,
            num_workers=2,
            preprocessed=True,
            pad_to_13_channels=False,  # Already padded
            val_split=0.1,
        )
        dm.setup()
        print(f"  Dataset loaded: OK")
        print(f"  Train samples: {len(dm.train_dataset)}")
        print(f"  Val samples: {len(dm.val_dataset)}")

        # Test a batch
        batch = next(iter(dm.train_dataloader()))
        print(f"  Batch image shape: {batch['image'].shape}")
        print(f"  Batch coords shape: {batch['coords'].shape}")
        print(f"  Image value range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")

        # Verify normalization
        if batch["image"].max() > 2.0:
            print("  WARNING: Image values seem unnormalized (max > 2.0)")
        else:
            print("  Normalization: OK (values in expected range)")

        return True

    except Exception as e:
        print(f"  Dataset loading: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vision_encoder():
    """Test vision encoder loading."""
    print("\n" + "=" * 60)
    print("Testing vision encoder...")
    print("=" * 60)

    import torch

    encoders = ["moco_resnet50", "moco_resnet18"]

    for encoder_name in encoders:
        try:
            from experiments.models.lightning_module import ContrastiveLearningModule

            model = ContrastiveLearningModule(
                vision_encoder=encoder_name,
                embed_dim=512,
                encoding_type="spherical_harmonics",
                encoding_config={"legendre_polys": 10},
                hidden_dim=512,
                num_layers=2,
                activation_type="siren",
            )

            # Test forward pass
            dummy_image = torch.randn(2, 13, 224, 224)
            dummy_coords = torch.randn(2, 2) * 90

            with torch.no_grad():
                loss = model.training_step(
                    {"image": dummy_image, "coords": dummy_coords}, 0
                )

            print(f"  {encoder_name}: OK (loss={loss['loss'].item():.4f})")

        except Exception as e:
            print(f"  {encoder_name}: FAILED - {e}")
            return False

    return True


def test_location_encoder():
    """Test location encoder with different configurations."""
    print("\n" + "=" * 60)
    print("Testing location encoder...")
    print("=" * 60)

    import torch
    from experiments.models.location_encoder import LocationEncoder

    configs = [
        {"encoding_type": "sh", "activation_type": "siren"},
        {"encoding_type": "sh", "activation_type": "relu"},
        {"encoding_type": "sh", "activation_type": "spline"},
        {"encoding_type": "cartesian3d", "activation_type": "siren"},
    ]

    coords = torch.randn(32, 2) * 90  # Random lon/lat

    for config in configs:
        try:
            encoder = LocationEncoder(**config)
            out = encoder(coords)
            config_str = f"{config['encoding_type']}+{config['activation_type']}"
            print(f"  {config_str}: OK (output shape={out.shape})")
        except Exception as e:
            print(f"  {config}: FAILED - {e}")
            return False

    return True


def test_eval_range():
    """Test RANGE evaluation framework."""
    print("\n" + "=" * 60)
    print("Testing RANGE evaluation...")
    print("=" * 60)

    try:
        from experiments.eval_range import (
            CheckerboardDataset,
            extract_embeddings,
            evaluate_embeddings,
        )

        # Test checkerboard dataset
        checker = CheckerboardDataset(n_samples=100, n_classes=4, n_support=50)
        print(f"  CheckerboardDataset: OK (train={len(checker.train_ds)}, test={len(checker.test_ds)})")

        # Test embedding extraction
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        class DummyEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(2, 64)

            def forward(self, x):
                return self.fc(x.float())

        encoder = DummyEncoder()
        coords = torch.randn(50, 2)
        labels = torch.randint(0, 4, (50,))
        loader = DataLoader(TensorDataset(coords, labels), batch_size=16)

        embeddings, labels_out = extract_embeddings(loader, encoder, device="cpu")
        print(f"  extract_embeddings: OK (shape={embeddings.shape})")

        return True

    except Exception as e:
        print(f"  RANGE evaluation: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mini_training(data_dir: str):
    """Test a minimal training loop."""
    print("\n" + "=" * 60)
    print("Testing mini training loop...")
    print("=" * 60)

    import torch
    import lightning.pytorch as pl

    # Use synthetic data for this test
    from experiments.models.lightning_module import LearnedActivationsModule
    from experiments.data import SyntheticDataModule

    try:
        datamodule = SyntheticDataModule(
            dataset_type="checkerboard",
            n_samples=100,
            batch_size=32,
            num_workers=0,
        )

        model = LearnedActivationsModule(
            encoding_type="sh",
            encoding_config={"legendre_polys": 5},
            activation_type="relu",
            hidden_dim=32,
            num_layers=2,
        )

        trainer = pl.Trainer(
            max_epochs=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
            accelerator="auto",
            devices=1,
        )

        trainer.fit(model, datamodule)
        print("  Synthetic training loop: OK")
        return True

    except Exception as e:
        print(f"  Training loop: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="HPC test suite for SatCLIP")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/projects/bdbk/cherd/data/satclip_manual",
        help="Path to data directory",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SatCLIP HPC Test Suite")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print()

    results = {}

    results["imports"] = test_imports()
    results["gpu"] = test_gpu()
    results["location_encoder"] = test_location_encoder()
    results["eval_range"] = test_eval_range()
    results["vision_encoder"] = test_vision_encoder()
    results["mini_training"] = test_mini_training(args.data_dir)

    # Dataset test (may be skipped if not preprocessed yet)
    dataset_result = test_preprocessed_dataset(args.data_dir)
    if dataset_result is not None:
        results["preprocessed_dataset"] = dataset_result

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests PASSED! Ready for training.")
    else:
        print("Some tests FAILED. Check output above for details.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
