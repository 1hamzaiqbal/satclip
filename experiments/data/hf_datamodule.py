"""HuggingFace Dataset-based DataModule for Sentinel-2 imagery.

Loads pre-processed data from HuggingFace Arrow format for faster training.
"""
import os
from typing import Optional, Dict, Any

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import pytorch_lightning as pl

from .coord_utils import coordinate_jitter


class HFSatelliteDataset(Dataset):
    """Wrapper around HuggingFace dataset for satellite imagery."""

    def __init__(
        self,
        hf_dataset,
        transform=None,
    ):
        """
        Args:
            hf_dataset: HuggingFace dataset (or subset) with 'image', 'lon', 'lat' columns.
            transform: Optional transform to apply to samples.
        """
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        item = self.dataset[idx]

        # With set_format("torch"), image is already a tensor
        image = item['image'].float()
        point = torch.tensor([item['lon'], item['lat']], dtype=torch.float32)

        sample = {"image": image, "point": point}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def get_hf_transform(
    crop_size: int = 256,
    jitter_radius: float = 0.01,
    pad_to_13_channels: bool = True,
    normalize: bool = True,
):
    """Get transform for HuggingFace satellite dataset.

    Args:
        crop_size: Output image size after cropping.
        jitter_radius: Radius for coordinate jitter augmentation.
        pad_to_13_channels: If True, insert zero-padded B10 channel at position 10.
        normalize: If True, divide by 10000 (raw Sentinel-2 values).
    """
    augmentation = T.Compose([
        T.RandomCrop(crop_size),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.GaussianBlur(kernel_size=3),
    ])

    def transform(sample: Dict[str, Any]) -> Dict[str, Tensor]:
        image = sample["image"]  # Already a tensor from set_format("torch")
        point = sample["point"]

        # Normalize: Sentinel-2 reflectance is stored as 0-10000
        if normalize:
            image = image / 10000.0

        # Optionally pad from 12 to 13 channels for pretrained model compatibility
        if pad_to_13_channels and image.shape[0] == 12:
            b10_zeros = torch.zeros((1, *image.shape[1:]), dtype=image.dtype)
            image = torch.cat([image[:10], b10_zeros, image[10:]], dim=0)

        image = augmentation(image)

        # Coordinate jitter for augmentation
        point = coordinate_jitter(point, radius=jitter_radius)

        # Keep as (lon, lat) - this is what the location encoders expect
        # point is already in (lon, lat) order from the dataset

        return {"image": image, "point": point}

    return transform


class HFMultispectralDataModule(pl.LightningDataModule):
    """DataModule for HuggingFace-based Sentinel-2 multispectral imagery.

    Much faster than loading individual TIF files since data is pre-loaded
    in memory-mapped Arrow format.
    """

    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 256,
        num_workers: int = 8,
        crop_size: int = 256,
        val_split: float = 0.1,
        pad_to_13_channels: bool = True,
        preprocessed: bool = False,
        seed: int = 42,
    ):
        """
        Args:
            dataset_path: Path to HuggingFace dataset (saved with save_to_disk).
            batch_size: Batch size for dataloaders.
            num_workers: Number of dataloader workers.
            crop_size: Output crop size (default 256 for Sentinel-2).
            val_split: Fraction of data to use for validation.
            pad_to_13_channels: If True, pad 12-channel images to 13 channels.
            seed: Random seed for train/val split.
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size
        self.val_split = val_split
        self.pad_to_13_channels = pad_to_13_channels
        self.preprocessed = preprocessed  # If True, skip normalization (already done)
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None
        self._num_channels = None

    @property
    def num_channels(self) -> int:
        """Return the number of output channels."""
        if self._num_channels is not None:
            return self._num_channels
        return 13 if self.pad_to_13_channels else 12

    def prepare_data(self):
        """Called only on rank 0. Nothing to prepare - data is pre-processed."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Initialize datasets."""
        from datasets import load_from_disk

        print(f"Loading HuggingFace dataset from: {self.dataset_path}")
        full_dataset = load_from_disk(self.dataset_path)
        full_dataset.set_format("torch")  # Return tensors directly, avoid numpy conversion
        print(f"Dataset size: {len(full_dataset)} samples")

        # Determine number of input channels from first sample
        # With set_format("torch"), this returns a tensor directly
        sample_image = full_dataset[0]['image']
        sample_shape = tuple(sample_image.shape)
        print(f"Image shape: {sample_shape}")
        input_channels = sample_shape[0]
        self._num_channels = 13 if (self.pad_to_13_channels and input_channels == 12) else input_channels

        # Split into train/val (format is preserved after split)
        split_dataset = full_dataset.train_test_split(
            test_size=self.val_split,
            seed=self.seed,
        )
        split_dataset['train'].set_format("torch")
        split_dataset['test'].set_format("torch")

        # Create transforms (skip normalization if data is preprocessed)
        train_transform = get_hf_transform(
            crop_size=self.crop_size,
            pad_to_13_channels=self.pad_to_13_channels,
            normalize=not self.preprocessed,
        )
        val_transform = get_hf_transform(
            crop_size=self.crop_size,
            jitter_radius=0.0,  # No jitter for validation
            pad_to_13_channels=self.pad_to_13_channels,
            normalize=not self.preprocessed,
        )

        # Wrap in PyTorch datasets
        self.train_dataset = HFSatelliteDataset(
            split_dataset['train'],
            transform=train_transform,
        )
        self.val_dataset = HFSatelliteDataset(
            split_dataset['test'],
            transform=val_transform,
        )

        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Output channels: {self.num_channels}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )


if __name__ == "__main__":
    # Test the HuggingFace datamodule
    print("Testing HFMultispectralDataModule...")
    print("=" * 60)

    data_module = HFMultispectralDataModule(
        dataset_path="/projects/bdbk/cherd/data/satclip_manual/satclip_hf",
        batch_size=32,
        num_workers=0,
        crop_size=256,
        pad_to_13_channels=True,
    )
    data_module.setup()

    # Get a sample batch
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))

    images = batch["image"]
    coords = batch["point"]

    print(f"\nBatch shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Coords: {coords.shape}")

    print(f"\nImage statistics (after /10000 normalization):")
    print(f"  Overall mean: {images.mean().item():.4f}")
    print(f"  Overall std:  {images.std().item():.4f}")
    print(f"  Min: {images.min().item():.4f}")
    print(f"  Max: {images.max().item():.4f}")

    # Check zero-padded channel
    if data_module.pad_to_13_channels:
        b10_channel = images[:, 10, :, :]
        print(f"\n  B10 (zero-padded) channel mean: {b10_channel.mean().item():.6f}")
        print(f"  B10 channel should be all zeros: {(b10_channel == 0).all().item()}")

    print(f"\nCoordinate statistics (lon, lat):")
    print(f"  Lon range: [{coords[:, 0].min().item():.2f}, {coords[:, 0].max().item():.2f}]")
    print(f"  Lat range: [{coords[:, 1].min().item():.2f}, {coords[:, 1].max().item():.2f}]")

    print("\n" + "=" * 60)
    print("Test complete!")
