#!/usr/bin/env python3
"""
Build a HuggingFace dataset from SatCLIP TIF files.

For fast training, this script stores images as numpy arrays directly in Arrow format,
which allows memory-mapped access without on-the-fly decoding overhead.

WHY ARROW/NUMPY IS FASTER THAN HDF5 FOR THIS USE CASE:
- Arrow format is memory-mapped, allowing random access without loading entire dataset
- HuggingFace DataLoader integrates seamlessly with Arrow
- No serialization overhead during training iteration
- Parallel data loading works out of the box
- HDF5 would require custom Dataset class and doesn't integrate as well with HF ecosystem

Usage:
    python build_hf_dataset.py --extract  # First extract tar.xz files
    python build_hf_dataset.py --build    # Then build the dataset
    python build_hf_dataset.py --extract --build  # Do both

Requirements:
    pip install datasets numpy pandas tqdm tifffile  # tifffile for multi-band TIF support
    # OR: pip install datasets numpy pandas tqdm rasterio  # rasterio as alternative
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from typing import Optional


def check_dependencies():
    """Check that required packages are installed."""
    missing = []
    for pkg in ['numpy', 'pandas', 'tqdm', 'datasets']:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    # Check for TIF reader
    tif_reader = None
    for pkg in ['tifffile', 'rasterio']:
        try:
            __import__(pkg)
            tif_reader = pkg
            break
        except ImportError:
            pass

    if tif_reader is None:
        missing.append('tifffile (or rasterio)')

    if missing:
        print("Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with: pip install " + " ".join(
            p.split()[0] for p in missing))
        sys.exit(1)

    return tif_reader


TIF_READER = check_dependencies()

import numpy as np
import pandas as pd
from tqdm import tqdm


def _extract_one_tar(args: tuple) -> str:
    """Extract a single tar.xz file. Module-level function for pickling."""
    tar_path, output_dir = args
    cmd = f"xz -dc '{tar_path}' | tar -xf - -C '{output_dir}'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        return f"Error extracting {tar_path.name}: {result.stderr}"
    return f"Extracted {tar_path.name}"


def extract_tar_files(data_dir: Path, output_dir: Path, num_workers: int = 4):
    """Extract all tar.xz files to output directory."""
    tar_files = sorted(data_dir.glob("images/data_*.tar.xz"))
    if not tar_files:
        # Check if tar files are in root directory
        tar_files = sorted(data_dir.glob("data_*.tar.xz"))

    if not tar_files:
        print("No tar.xz files found!")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(tar_files)} tar.xz files to extract")

    # Use ThreadPoolExecutor (subprocess calls are I/O bound, no pickling issues)
    extract_args = [(f, output_dir) for f in tar_files]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(_extract_one_tar, extract_args),
            total=len(tar_files),
            desc="Extracting"
        ))

    for result in results:
        if "Error" in result:
            print(f"\n{result}")


def load_tif_file(filepath: Path) -> Optional[np.ndarray]:
    """Load a TIF file and return as numpy array.

    Tries multiple backends for compatibility.
    Returns array in (C, H, W) format.
    Returns None for corrupted/unreadable files.
    """
    try:
        # Try tifffile first (best for multi-band GeoTIFFs)
        import tifffile
        img = tifffile.imread(str(filepath))
        # tifffile returns (H, W) for single band, (H, W, C) or (C, H, W) for multi-band
        if img.ndim == 2:
            img = img[np.newaxis, :, :]  # Add channel dim
        elif img.ndim == 3:
            # Check if channels are last (most common) or first
            if img.shape[2] <= 13:  # Likely (H, W, C)
                img = np.transpose(img, (2, 0, 1))
            # else assume (C, H, W)
        return img.astype(np.float32)
    except ImportError:
        pass
    except Exception:
        # Corrupted file, try next backend
        pass

    try:
        # Try rasterio (good for geospatial data)
        import rasterio
        with rasterio.open(filepath) as src:
            img = src.read()  # Returns (C, H, W)
        return img.astype(np.float32)
    except ImportError:
        pass
    except Exception:
        pass

    try:
        # Fallback to PIL (only works for RGB)
        from PIL import Image
        img = np.array(Image.open(filepath))
        if img.ndim == 2:
            img = img[np.newaxis, :, :]
        elif img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))
        return img.astype(np.float32)
    except Exception:
        pass

    # All backends failed - file is corrupted or unsupported
    return None


def build_dataset(
    data_dir: Path,
    tif_dir: Path,
    output_path: Path,
    num_workers: int = 8,
):
    """Build HuggingFace dataset from extracted TIF files.

    Uses generator-based approach for memory efficiency - processes one image
    at a time instead of loading all into memory.
    """
    from datasets import Dataset, Features, Value, Array3D

    # Load index CSV
    index_path = data_dir / "index.csv"
    if not index_path.exists():
        print(f"Index file not found: {index_path}")
        return

    df = pd.read_csv(index_path)
    print(f"Index contains {len(df)} entries")

    # Find which TIF files actually exist
    existing_files = set(f.name for f in tif_dir.glob("*.tif"))
    print(f"Found {len(existing_files)} TIF files on disk")

    # Filter to existing files
    df_filtered = df[df['fn'].isin(existing_files)].copy()
    print(f"Matched {len(df_filtered)} entries with existing files")

    if len(df_filtered) == 0:
        print("No matching files found!")
        return

    # Determine image shape from first file
    sample_file = tif_dir / df_filtered.iloc[0]['fn']
    sample_img = load_tif_file(sample_file)
    if sample_img is None:
        print("Could not load sample file to determine shape")
        return

    channels, height, width = sample_img.shape
    print(f"Image shape: {channels} channels, {height}x{width} pixels")

    # Create list of file info for generator
    file_info = [
        (tif_dir / row['fn'], row['lon'], row['lat'], row['fn'])
        for _, row in df_filtered.iterrows()
    ]

    # Track statistics
    stats = {'processed': 0, 'skipped': 0}

    def data_generator():
        """Generator that yields one sample at a time - memory efficient."""
        for filepath, lon, lat, filename in tqdm(file_info, desc="Building dataset"):
            img = load_tif_file(filepath)
            if img is not None:
                stats['processed'] += 1
                yield {
                    'image': img,
                    'lon': lon,
                    'lat': lat,
                    'filename': filename,
                }
            else:
                stats['skipped'] += 1

    # Create HuggingFace dataset with proper features
    features = Features({
        'image': Array3D(shape=(channels, height, width), dtype='float32'),
        'lon': Value('float64'),
        'lat': Value('float64'),
        'filename': Value('string'),
    })

    print("Creating HuggingFace dataset (streaming to disk)...")
    dataset = Dataset.from_generator(
        data_generator,
        features=features,
        num_proc=1,  # Generator is already sequential
    )

    print(f"Successfully processed {stats['processed']} images")
    if stats['skipped'] > 0:
        print(f"Skipped {stats['skipped']} corrupted/unreadable files")

    # Save to disk
    print(f"Saving dataset to {output_path}...")
    dataset.save_to_disk(str(output_path))

    print(f"\nDataset created successfully!")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Image shape: ({channels}, {height}, {width})")
    print(f"  Location: {output_path}")
    print(f"\n" + "=" * 60)
    print("USAGE EXAMPLE FOR TRAINING:")
    print("=" * 60)
    print(f"""
from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader

# Load dataset
dataset = load_from_disk('{output_path}')

# Optional: split into train/val
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = dataset['train']
val_ds = dataset['test']

# Convert to PyTorch format for fast training
train_ds.set_format(type='torch', columns=['image', 'lon', 'lat'])

# Create DataLoader
def collate_fn(batch):
    images = torch.stack([x['image'] for x in batch])
    lons = torch.tensor([x['lon'] for x in batch])
    lats = torch.tensor([x['lat'] for x in batch])
    return {{'image': images, 'lon': lons, 'lat': lats}}

train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True,
)

# Training loop
for batch in train_loader:
    images = batch['image']  # (B, C, H, W) tensor
    coords = torch.stack([batch['lon'], batch['lat']], dim=1)  # (B, 2)
    # ... your training code here
""")

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Build HuggingFace dataset from SatCLIP TIF files")
    parser.add_argument("--data-dir", type=Path, default=Path("/projects/bdbk/cherd/data/satclip_manual"),
                        help="Directory containing index.csv and tar.xz files")
    parser.add_argument("--tif-dir", type=Path, default=None,
                        help="Directory containing extracted TIF files (default: data_dir/tif)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path for HuggingFace dataset (default: data_dir/satclip_hf)")
    parser.add_argument("--extract", action="store_true",
                        help="Extract tar.xz files")
    parser.add_argument("--build", action="store_true",
                        help="Build HuggingFace dataset")
    parser.add_argument("--workers", type=int, default=min(8, multiprocessing.cpu_count()),
                        help="Number of worker processes")

    args = parser.parse_args()

    # Set defaults
    if args.tif_dir is None:
        args.tif_dir = args.data_dir / "tif"
    if args.output is None:
        args.output = args.data_dir / "satclip_hf"

    if not args.extract and not args.build:
        parser.print_help()
        print("\nError: Specify at least one of --extract or --build")
        sys.exit(1)

    if args.extract:
        print("=" * 60)
        print("STEP 1: Extracting tar.xz files")
        print("=" * 60)
        extract_tar_files(args.data_dir, args.tif_dir, args.workers)

    if args.build:
        print("\n" + "=" * 60)
        print("STEP 2: Building HuggingFace dataset")
        print("=" * 60)
        build_dataset(args.data_dir, args.tif_dir, args.output, args.workers)


if __name__ == "__main__":
    main()
