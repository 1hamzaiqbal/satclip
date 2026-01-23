#!/usr/bin/env python
"""Test script to verify the experiments setup works correctly.

Run this before submitting HPC jobs to catch issues early.

Usage:
    python experiments/scripts/local/test_setup.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from experiments.models.activations import (
            ReLUActivation,
            SIRENActivation,
            SplineActivation,
            RFFActivation,
            get_activation,
        )
        print("  Activations: OK")
    except ImportError as e:
        print(f"  Activations: FAILED - {e}")
        return False

    try:
        from experiments.models.encodings import (
            SphericalHarmonicsEncoding,
            RawCoordinateEncoding,
            RFFEncoding,
            get_encoding,
        )
        print("  Encodings: OK")
    except ImportError as e:
        print(f"  Encodings: FAILED - {e}")
        return False

    try:
        from experiments.models.location_encoder import (
            LocationEncoder,
            LocationRegressor,
        )
        print("  Location Encoder: OK")
    except ImportError as e:
        print(f"  Location Encoder: FAILED - {e}")
        return False

    try:
        from experiments.models.lightning_module import (
            LearnedActivationsModule,
        )
        print("  Lightning Module: OK")
    except ImportError as e:
        print(f"  Lightning Module: FAILED - {e}")
        return False

    try:
        from experiments.data import (
            CheckerboardDataset,
            SyntheticDataModule,
        )
        print("  Data: OK")
    except ImportError as e:
        print(f"  Data: FAILED - {e}")
        return False

    # New: test HF datamodule and coord_utils imports
    try:
        from experiments.data import (
            HFMultispectralDataModule,
            HFSatelliteDataset,
            latlon_to_sphere,
            sphere_to_latlon,
            coordinate_jitter,
            load_sites,
        )
        print("  HF Datamodule & Coord Utils: OK")
    except ImportError as e:
        print(f"  HF Datamodule & Coord Utils: FAILED - {e}")
        return False

    return True


def test_activations():
    """Test activation functions."""
    print("\nTesting activations...")
    import torch
    from experiments.models.activations import get_activation

    x = torch.randn(32, 256)

    for name in ["relu", "spline"]:
        try:
            act = get_activation(name)
            out = act(x)
            assert out.shape == x.shape
            print(f"  {name}: OK (shape={out.shape})")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            return False

    return True


def test_encodings():
    """Test positional encodings."""
    print("\nTesting encodings...")
    import torch
    from experiments.models.encodings import get_encoding

    coords = torch.randn(32, 2) * 90  # Random lon/lat

    for name in ["sh", "raw", "cartesian3d"]:
        try:
            enc = get_encoding(name)
            out = enc(coords)
            print(f"  {name}: OK (input={coords.shape}, output={out.shape})")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            return False

    return True


def test_location_encoder():
    """Test location encoder."""
    print("\nTesting location encoder...")
    import torch
    from experiments.models.location_encoder import LocationEncoder

    coords = torch.randn(32, 2) * 90

    configs = [
        {"encoding_type": "sh", "activation_type": "relu"},
        {"encoding_type": "raw", "activation_type": "spline"},
    ]

    for config in configs:
        try:
            encoder = LocationEncoder(**config)
            out = encoder(coords)
            print(f"  {config['encoding_type']}+{config['activation_type']}: OK (output={out.shape})")
        except Exception as e:
            print(f"  {config}: FAILED - {e}")
            return False

    return True


def test_synthetic_data():
    """Test synthetic data loading."""
    print("\nTesting synthetic data...")
    from experiments.data import CheckerboardDataset

    try:
        dataset = CheckerboardDataset(n_samples=100)
        sample = dataset[0]
        print(f"  CheckerboardDataset: OK (coords={sample['coords'].shape}, target={sample['target'].shape})")
    except Exception as e:
        print(f"  CheckerboardDataset: FAILED - {e}")
        return False

    return True


def test_lightning_module():
    """Test Lightning module instantiation."""
    print("\nTesting Lightning module...")
    from experiments.models.lightning_module import LearnedActivationsModule

    try:
        module = LearnedActivationsModule(
            encoding_type="sh",
            activation_type="relu",
            hidden_dim=64,
            num_layers=2,
        )
        print(f"  LearnedActivationsModule: OK")
    except Exception as e:
        print(f"  LearnedActivationsModule: FAILED - {e}")
        return False

    return True


def test_coord_utils():
    """Test coordinate utility functions."""
    print("\nTesting coordinate utilities...")
    import torch
    from experiments.data import latlon_to_sphere, sphere_to_latlon, coordinate_jitter

    try:
        # Test lat/lon to sphere conversion
        coords = torch.tensor([[37.7749, -122.4194], [40.7128, -74.0060]])  # SF, NYC
        xyz = latlon_to_sphere(coords)
        assert xyz.shape == (2, 3), f"Expected shape (2, 3), got {xyz.shape}"

        # Verify unit sphere (norm should be ~1)
        norms = torch.norm(xyz, dim=-1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5), "Points not on unit sphere"
        print("  latlon_to_sphere: OK")

        # Test sphere to lat/lon conversion (round trip)
        coords_back = sphere_to_latlon(xyz)
        assert torch.allclose(coords, coords_back, atol=1e-4), "Round trip failed"
        print("  sphere_to_latlon: OK")

        # Test coordinate jitter
        jittered = coordinate_jitter(coords, radius=0.01)
        assert jittered.shape == coords.shape, "Jitter changed shape"
        assert not torch.allclose(jittered, coords), "Jitter didn't change coordinates"
        print("  coordinate_jitter: OK")

        return True

    except Exception as e:
        print(f"  Coordinate utilities: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_eval_range():
    """Test RANGE evaluation framework."""
    print("\nTesting RANGE evaluation framework...")

    try:
        from experiments.eval_range import (
            CheckerboardDataset,
            extract_embeddings,
            evaluate_embeddings,
            get_dataset,
        )
        print("  eval_range imports: OK")

        # Test checkerboard dataset (synthetic, no files needed)
        checker = CheckerboardDataset(n_samples=100, n_classes=4, n_support=50)
        assert len(checker.train_ds) == 100, "Wrong train size"
        print("  CheckerboardDataset: OK")

        # Test extraction with a simple encoder
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        class DummyEncoder(nn.Module):
            def __init__(self, output_dim=64):
                super().__init__()
                self.fc = nn.Linear(2, output_dim)

            def forward(self, x):
                return self.fc(x.float())

        encoder = DummyEncoder()
        coords = torch.randn(50, 2)
        labels = torch.randint(0, 4, (50,))
        loader = DataLoader(TensorDataset(coords, labels), batch_size=16)

        embeddings, labels_out = extract_embeddings(loader, encoder, device="cpu")
        assert embeddings.shape == (50, 64), f"Expected (50, 64), got {embeddings.shape}"
        print("  extract_embeddings: OK")

        return True

    except Exception as e:
        print(f"  RANGE evaluation: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_training_loop():
    """Test a minimal training loop."""
    print("\nTesting full training loop (1 epoch, 10 samples)...")
    import torch
    import lightning.pytorch as pl
    from experiments.models.lightning_module import LearnedActivationsModule
    from experiments.data import SyntheticDataModule

    try:
        # Tiny experiment
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
        )

        trainer.fit(model, datamodule)
        print("  Full training loop: OK")
        return True

    except Exception as e:
        print(f"  Full training loop: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Learned Activations Setup Test")
    print("=" * 60)

    all_passed = True

    all_passed &= test_imports()
    all_passed &= test_activations()
    all_passed &= test_encodings()
    all_passed &= test_location_encoder()
    all_passed &= test_synthetic_data()
    all_passed &= test_coord_utils()
    all_passed &= test_eval_range()
    all_passed &= test_lightning_module()
    all_passed &= test_full_training_loop()

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED! Ready to submit HPC jobs.")
        print("\nNext steps:")
        print("  1. Submit preprocessing job to HPC:")
        print("     sbatch experiments/scripts/slurm/preprocess_s2k.sh")
        print("")
        print("  2. Once preprocessing is done, run training:")
        print("     sbatch experiments/scripts/slurm/submit_contrastive.sh")
        print("     # Or for multispectral:")
        print("     python -m experiments.train --config experiments/configs/experiments/contrastive_multispectral.yaml")
        print("")
        print("  3. Evaluate trained model:")
        print("     sbatch experiments/scripts/slurm/eval_range.sh --model_path /path/to/checkpoint.pt")
    else:
        print("Some tests FAILED. Fix issues before submitting jobs.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
