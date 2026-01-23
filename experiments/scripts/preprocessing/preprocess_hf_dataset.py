#!/usr/bin/env python3
"""
Pre-process HuggingFace satellite dataset for faster training.

This script applies expensive preprocessing operations ONCE instead of
on every sample access during training:
- Normalize by dividing by 10000 (Sentinel-2 reflectance scale)
- Pad from 12 to 13 channels (insert zero B10 at position 10)
- Optionally center crop to target size

This can provide 2-3x speedup in data loading during training.

Usage:
    # Pre-process with default settings (normalize + pad, keep 256x256)
    python preprocess_hf_dataset.py

    # Pre-process and center crop to 224x224
    python preprocess_hf_dataset.py --crop 224

    # Custom input/output paths
    python preprocess_hf_dataset.py --input /path/to/dataset --output /path/to/output

    # Skip normalization (if you want raw values)
    python preprocess_hf_dataset.py --no-normalize
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


def preprocess_sample(
    sample: dict,
    normalize: bool = True,
    pad_to_13_channels: bool = True,
    crop_size: int = None,
) -> dict:
    """
    Pre-apply normalization, padding, and optional cropping.

    Args:
        sample: Dict with 'image', 'lon', 'lat', 'filename' keys
        normalize: If True, divide by 10000 (Sentinel-2 scale)
        pad_to_13_channels: If True, insert zero B10 channel at position 10
        crop_size: If set, center crop to this size (e.g., 224)

    Returns:
        Preprocessed sample dict
    """
    # Get image as numpy array
    image = np.array(sample['image'], dtype=np.float32)

    # Normalize: Sentinel-2 reflectance is stored as 0-10000
    if normalize:
        image = image / 10000.0

    # Pad to 13 channels (insert B10 zeros at position 10)
    if pad_to_13_channels and image.shape[0] == 12:
        b10_zeros = np.zeros((1, *image.shape[1:]), dtype=image.dtype)
        image = np.concatenate([image[:10], b10_zeros, image[10:]], axis=0)

    # Center crop if requested
    if crop_size is not None:
        _, h, w = image.shape
        if h >= crop_size and w >= crop_size:
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
            image = image[:, top:top+crop_size, left:left+crop_size]
        else:
            print(f"Warning: Image size ({h}x{w}) smaller than crop size ({crop_size})")

    return {
        'image': image,
        'lon': sample['lon'],
        'lat': sample['lat'],
        'filename': sample['filename'],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pre-process HuggingFace satellite dataset for faster training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default: normalize + pad to 13 channels, keep original size
    python preprocess_hf_dataset.py

    # Also center crop to 224x224
    python preprocess_hf_dataset.py --crop 224

    # Custom paths
    python preprocess_hf_dataset.py \\
        --input /path/to/satclip_hf \\
        --output /path/to/satclip_hf_preprocessed
        """
    )
    parser.add_argument(
        "--input", type=Path,
        default=Path("/projects/bdbk/cherd/data/satclip_manual/satclip_hf"),
        help="Path to input HuggingFace dataset"
    )
    parser.add_argument(
        "--output", type=Path,
        default=None,
        help="Path for output dataset (default: input_preprocessed)"
    )
    parser.add_argument(
        "--crop", type=int, default=None,
        help="Center crop to this size (e.g., 224). If not set, keep original size."
    )
    parser.add_argument(
        "--no-normalize", action="store_true",
        help="Skip normalization (keep raw 0-10000 values)"
    )
    parser.add_argument(
        "--no-pad", action="store_true",
        help="Skip padding to 13 channels"
    )
    parser.add_argument(
        "--num-proc", type=int, default=8,
        help="Number of processes for parallel processing"
    )

    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        suffix = "_preprocessed"
        if args.crop:
            suffix += f"_{args.crop}"
        args.output = Path(str(args.input) + suffix)

    # Check input exists
    if not args.input.exists():
        print(f"Error: Input dataset not found: {args.input}")
        sys.exit(1)

    # Import HuggingFace datasets
    try:
        from datasets import load_from_disk, Dataset, Features, Value, Array3D
    except ImportError:
        print("Error: Please install datasets: pip install datasets")
        sys.exit(1)

    # Load input dataset
    print(f"Loading dataset from: {args.input}")
    dataset = load_from_disk(str(args.input))
    print(f"Dataset size: {len(dataset)} samples")

    # Get original shape
    sample = dataset[0]
    orig_image = np.array(sample['image'])
    orig_channels, orig_h, orig_w = orig_image.shape
    print(f"Original image shape: {orig_channels} channels, {orig_h}x{orig_w}")

    # Calculate output shape
    out_channels = 13 if (not args.no_pad and orig_channels == 12) else orig_channels
    out_h = args.crop if args.crop else orig_h
    out_w = args.crop if args.crop else orig_w

    print(f"\nPreprocessing settings:")
    print(f"  Normalize: {not args.no_normalize}")
    print(f"  Pad to 13 channels: {not args.no_pad}")
    print(f"  Center crop: {args.crop if args.crop else 'None (keep original)'}")
    print(f"  Output shape: {out_channels} channels, {out_h}x{out_w}")
    print(f"  Output path: {args.output}")
    print()

    # Process using map for efficiency
    def process_batch(batch):
        """Process a batch of samples."""
        processed_images = []
        for i in range(len(batch['image'])):
            sample = {
                'image': batch['image'][i],
                'lon': batch['lon'][i],
                'lat': batch['lat'][i],
                'filename': batch['filename'][i],
            }
            result = preprocess_sample(
                sample,
                normalize=not args.no_normalize,
                pad_to_13_channels=not args.no_pad,
                crop_size=args.crop,
            )
            processed_images.append(result['image'])

        return {
            'image': processed_images,
            'lon': batch['lon'],
            'lat': batch['lat'],
            'filename': batch['filename'],
        }

    print("Processing dataset...")

    # Use map with batched=True for efficiency
    processed_dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=100,
        num_proc=args.num_proc,
        desc="Preprocessing",
    )

    # Cast to proper features for optimal storage
    features = Features({
        'image': Array3D(shape=(out_channels, out_h, out_w), dtype='float32'),
        'lon': Value('float64'),
        'lat': Value('float64'),
        'filename': Value('string'),
    })

    processed_dataset = processed_dataset.cast(features)

    # Save to disk
    print(f"\nSaving to: {args.output}")
    processed_dataset.save_to_disk(str(args.output))

    # Print summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Samples: {len(processed_dataset)}")
    print(f"  Shape: ({out_channels}, {out_h}, {out_w})")

    # Verify a sample
    print("\nVerifying output...")
    test_sample = processed_dataset[0]
    test_img = np.array(test_sample['image'])
    print(f"  Sample shape: {test_img.shape}")
    print(f"  Sample dtype: {test_img.dtype}")
    print(f"  Value range: [{test_img.min():.4f}, {test_img.max():.4f}]")

    if not args.no_pad and out_channels == 13:
        b10 = test_img[10]
        print(f"  B10 channel (should be zeros): mean={b10.mean():.6f}, max={b10.max():.6f}")

    print("\n" + "=" * 60)
    print("USAGE IN TRAINING CONFIG:")
    print("=" * 60)
    print(f"""
# In configs/base_multispectral.yaml, update:
data:
  hf_dataset_path: "{args.output}"
  pad_to_13_channels: false  # Already padded!

# And in hf_datamodule.py, the transform can skip normalization
# since data is already normalized.
""")

    if args.crop:
        print(f"""
NOTE: Since images are pre-cropped to {args.crop}x{args.crop}, you may want to:
1. Remove RandomCrop from augmentations (or use smaller crop)
2. Or keep 256x256 and use RandomCrop for augmentation diversity
""")


if __name__ == "__main__":
    main()
