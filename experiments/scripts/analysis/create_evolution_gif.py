#!/usr/bin/env python
"""Create animated GIF from epoch visualizations.

Combines visualization images into an animated GIF showing training evolution.

Usage:
    # From generated epoch images
    python -m experiments.scripts.analysis.create_evolution_gif \
        --input_dir epoch_visualizations \
        --output training_evolution.gif

    # Custom settings
    python -m experiments.scripts.analysis.create_evolution_gif \
        --input_dir epoch_visualizations \
        --output training_evolution.gif \
        --pattern "combined_epoch_*.png" \
        --duration 500 \
        --loop

    # Create GIF for globe only
    python -m experiments.scripts.analysis.create_evolution_gif \
        --input_dir epoch_visualizations \
        --output globe_evolution.gif \
        --pattern "globe_epoch_*.png"
"""

import argparse
import re
from pathlib import Path
from typing import List

from PIL import Image


def get_epoch_from_filename(filename: str) -> int:
    """Extract epoch number from filename."""
    patterns = [
        r'epoch[_=]?(\d+)',
        r'_(\d+)\.',
    ]
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return -1


def create_gif(
    image_paths: List[Path],
    output_path: str,
    duration: int = 500,
    loop: bool = True,
):
    """Create animated GIF from image sequence.

    Args:
        image_paths: List of paths to image files (in order)
        output_path: Output GIF path
        duration: Duration per frame in milliseconds
        loop: Whether to loop (0=infinite loop, else number of loops)
    """
    if not image_paths:
        print("No images to process")
        return

    # Load all images
    images = []
    for path in image_paths:
        try:
            img = Image.open(path)
            # Convert to RGB if necessary (RGBA can cause issues)
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")

    if not images:
        print("No valid images loaded")
        return

    print(f"Creating GIF from {len(images)} frames...")

    # Save as GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0 if loop else 1,  # 0 means infinite loop
        optimize=True,
    )

    print(f"Saved: {output_path}")
    print(f"  Frames: {len(images)}")
    print(f"  Duration: {duration}ms per frame")
    print(f"  Total: {len(images) * duration / 1000:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Create animated GIF from epoch visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--input_dir', '-i',
        type=str,
        required=True,
        help='Directory containing visualization images',
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='training_evolution.gif',
        help='Output GIF path',
    )
    parser.add_argument(
        '--pattern', '-p',
        type=str,
        default='combined_epoch_*.png',
        help='Glob pattern for input images (default: combined_epoch_*.png)',
    )
    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=500,
        help='Duration per frame in milliseconds (default: 500)',
    )
    parser.add_argument(
        '--no_loop',
        action='store_true',
        help='Do not loop the GIF',
    )
    parser.add_argument(
        '--reverse',
        action='store_true',
        help='Add reverse playback (ping-pong effect)',
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}")
        return

    # Find matching images
    image_paths = list(input_dir.glob(args.pattern))

    if not image_paths:
        print(f"No images found matching pattern: {args.pattern}")
        print("Available files:")
        for f in sorted(input_dir.glob('*'))[:10]:
            print(f"  {f.name}")
        return

    # Sort by epoch
    image_paths = sorted(image_paths, key=lambda p: get_epoch_from_filename(p.name))

    print(f"Found {len(image_paths)} images")
    print(f"  First: {image_paths[0].name}")
    print(f"  Last: {image_paths[-1].name}")

    # Add reverse for ping-pong effect
    if args.reverse:
        image_paths = image_paths + image_paths[-2:0:-1]
        print(f"  With reverse: {len(image_paths)} frames")

    # Create GIF
    create_gif(
        image_paths,
        args.output,
        duration=args.duration,
        loop=not args.no_loop,
    )


if __name__ == '__main__':
    main()
