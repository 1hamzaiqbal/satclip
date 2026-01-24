#!/usr/bin/env python
"""Generate visualizations for each training epoch from checkpoints.

Extracts location encoder state from checkpoints and generates:
1. Globe embeddings visualization (PCA → RGB)
2. Spline activation shapes (if using spline activation)
3. Combined dashboard view

Usage:
    # Visualize all checkpoints
    python -m experiments.scripts.analysis.visualize_epochs \
        --checkpoint_dir /path/to/checkpoints \
        --output_dir epoch_visualizations

    # Specific epochs only
    python -m experiments.scripts.analysis.visualize_epochs \
        --checkpoint_dir /path/to/checkpoints \
        --output_dir epoch_visualizations \
        --epochs 0 10 50

    # Globe only (faster)
    python -m experiments.scripts.analysis.visualize_epochs \
        --checkpoint_dir /path/to/checkpoints \
        --output_dir epoch_visualizations \
        --globe_only
"""

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.models.lightning_module import ContrastiveLearningModule
from experiments.models.activations.spline import SplineActivation

# Try to import cartopy
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Warning: cartopy not installed. Using basic projection.")

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def create_globe_grid(lat_res: int = 90, lon_res: int = 180):
    """Create a grid of lat/lon points covering the globe."""
    lats = np.linspace(-90, 90, lat_res)
    lons = np.linspace(-180, 180, lon_res)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    coords = np.stack([lon_grid.flatten(), lat_grid.flatten()], axis=-1)
    return lats, lons, coords


def get_epoch_from_filename(filename: str) -> int:
    """Extract epoch number from checkpoint filename."""
    # Try different patterns
    patterns = [
        r'epoch[_=](\d+)',
        r'epoch(\d+)',
        r'-(\d+)-',  # Lightning default format
    ]
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return -1


def get_val_loss_from_filename(filename: str) -> Optional[float]:
    """Extract validation loss from checkpoint filename."""
    match = re.search(r'val_loss[_=]?(\d+\.?\d*)', filename)
    if match:
        return float(match.group(1))
    return None


def find_spline_activations(model) -> List[Tuple[str, SplineActivation]]:
    """Find all SplineActivation modules in the model."""
    splines = []
    for name, module in model.named_modules():
        if isinstance(module, SplineActivation):
            splines.append((name, module))
    return splines


def encode_globe(model, coords_tensor: torch.Tensor, device: str = 'cpu') -> np.ndarray:
    """Encode globe grid using location encoder."""
    coords = coords_tensor.to(device)
    batch_size = 4096
    embeddings = []

    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            batch = coords[i:i + batch_size]
            emb = model.encode_location(batch)
            embeddings.append(emb.cpu())

    return torch.cat(embeddings, dim=0).numpy()


def embeddings_to_colors(embeddings: np.ndarray) -> np.ndarray:
    """Convert embeddings to RGB colors using PCA."""
    if HAS_SKLEARN:
        pca = PCA(n_components=3)
        colors = pca.fit_transform(embeddings)
    else:
        colors = embeddings[:, :3].copy()

    colors = colors - colors.min(axis=0)
    colors = colors / (colors.max(axis=0) + 1e-8)
    return colors


def plot_globe_cartopy(
    colors: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    lat_res: int,
    lon_res: int,
    title: str,
    output_path: str,
):
    """Plot globe visualization using cartopy."""
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')

    ax.scatter(
        lon_grid.flatten(), lat_grid.flatten(),
        c=colors, s=3, alpha=0.8,
        transform=ccrs.PlateCarree(), marker='s',
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5, zorder=3)

    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_globe_basic(
    colors: np.ndarray,
    lat_res: int,
    lon_res: int,
    title: str,
    output_path: str,
):
    """Plot globe visualization using basic matplotlib."""
    color_grid = colors.reshape(lat_res, lon_res, 3)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(color_grid, extent=[-180, 180, -90, 90], origin='lower', aspect='equal')

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_splines(
    splines: List[Tuple[str, SplineActivation]],
    title: str,
    output_path: str,
):
    """Plot spline activations."""
    n_splines = len(splines)
    if n_splines == 0:
        return

    n_cols = min(3, n_splines)
    n_rows = (n_splines + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_splines == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    x = torch.linspace(-4, 4, 200)
    x_np = x.numpy()

    for idx, (name, spline) in enumerate(splines):
        ax = axes[idx]
        knot_x, knot_y = spline.get_knot_data()

        with torch.no_grad():
            y = spline(x.to(knot_x.device)).cpu()

        ax.plot(x_np, y.numpy(), 'b-', linewidth=2, label='Learned')
        ax.scatter(knot_x.numpy(), knot_y.numpy(), c='red', s=30, zorder=5)
        ax.plot(x_np, np.maximum(0, x_np), 'g--', alpha=0.4, label='ReLU')

        short_name = name.split('.')[-2] if '.' in name else name
        ax.set_title(f'Layer: {short_name}')
        ax.set_xlabel('Input')
        ax.set_ylabel('Output')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-4, 4)

    for idx in range(n_splines, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_combined(
    model,
    colors: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    lat_res: int,
    lon_res: int,
    splines: List[Tuple[str, SplineActivation]],
    title: str,
    output_path: str,
):
    """Plot combined dashboard with splines and globe."""
    has_splines = len(splines) > 0

    if has_splines:
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5])
        ax_spline = fig.add_subplot(gs[0])
        if HAS_CARTOPY:
            ax_globe = fig.add_subplot(gs[1], projection=ccrs.Robinson())
        else:
            ax_globe = fig.add_subplot(gs[1])
    else:
        fig = plt.figure(figsize=(12, 6))
        if HAS_CARTOPY:
            ax_globe = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        else:
            ax_globe = fig.add_subplot(1, 1, 1)
        ax_spline = None

    # Plot splines
    if ax_spline is not None and has_splines:
        x = torch.linspace(-4, 4, 200)
        colors_spline = plt.cm.viridis(np.linspace(0, 1, len(splines)))

        for (name, spline), color in zip(splines, colors_spline):
            knot_x, knot_y = spline.get_knot_data()
            with torch.no_grad():
                y = spline(x.to(knot_x.device)).cpu()

            short_name = name.split('.')[-2] if '.' in name else name
            ax_spline.plot(x.numpy(), y.numpy(), color=color, linewidth=2, label=short_name)
            ax_spline.scatter(knot_x.numpy(), knot_y.numpy(), c=[color], s=20, zorder=5)

        ax_spline.plot(x.numpy(), np.maximum(0, x.numpy()), 'k--', alpha=0.3, label='ReLU')
        ax_spline.set_xlabel('Input')
        ax_spline.set_ylabel('Output')
        ax_spline.set_title('Spline Activations')
        ax_spline.legend(fontsize=8)
        ax_spline.grid(True, alpha=0.3)
        ax_spline.set_xlim(-4, 4)

    # Plot globe
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')

    if HAS_CARTOPY and hasattr(ax_globe, 'set_global'):
        ax_globe.set_global()
        ax_globe.scatter(
            lon_grid.flatten(), lat_grid.flatten(),
            c=colors, s=2, alpha=0.8,
            transform=ccrs.PlateCarree(), marker='s',
        )
        ax_globe.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    else:
        color_grid = colors.reshape(lat_res, lon_res, 3)
        ax_globe.imshow(color_grid, extent=[-180, 180, -90, 90], origin='lower', aspect='equal')
        ax_globe.set_xlabel('Longitude')
        ax_globe.set_ylabel('Latitude')

    ax_globe.set_title('Location Embeddings (PCA → RGB)')

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_checkpoint(
    ckpt_path: Path,
    output_dir: Path,
    lats: np.ndarray,
    lons: np.ndarray,
    coords_tensor: torch.Tensor,
    lat_res: int,
    lon_res: int,
    globe_only: bool = False,
    splines_only: bool = False,
    combined: bool = True,
):
    """Process a single checkpoint and generate visualizations."""
    epoch = get_epoch_from_filename(ckpt_path.name)
    val_loss = get_val_loss_from_filename(ckpt_path.name)

    print(f"  Processing epoch {epoch}...")

    # Load model
    try:
        model = ContrastiveLearningModule.load_from_checkpoint(
            str(ckpt_path),
            map_location='cpu',
        )
        model.eval()
    except Exception as e:
        print(f"    Error loading checkpoint: {e}")
        return

    # Get splines
    splines = find_spline_activations(model)
    has_splines = len(splines) > 0

    # Encode globe
    embeddings = encode_globe(model, coords_tensor)
    colors = embeddings_to_colors(embeddings)

    # Create title
    loss_str = f", val_loss={val_loss:.4f}" if val_loss else ""
    title_base = f"Epoch {epoch}{loss_str}"

    # Save globe visualization
    if not splines_only:
        globe_path = output_dir / f"globe_epoch_{epoch:03d}.png"
        if HAS_CARTOPY:
            plot_globe_cartopy(colors, lats, lons, lat_res, lon_res, title_base, str(globe_path))
        else:
            plot_globe_basic(colors, lat_res, lon_res, title_base, str(globe_path))
        print(f"    Saved: {globe_path.name}")

    # Save spline visualization
    if has_splines and not globe_only:
        spline_path = output_dir / f"splines_epoch_{epoch:03d}.png"
        plot_splines(splines, title_base, str(spline_path))
        print(f"    Saved: {spline_path.name}")

    # Save combined visualization
    if combined and (has_splines or not globe_only):
        combined_path = output_dir / f"combined_epoch_{epoch:03d}.png"
        plot_combined(model, colors, lats, lons, lat_res, lon_res, splines, title_base, str(combined_path))
        print(f"    Saved: {combined_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations for training epochs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--checkpoint_dir', '-c',
        type=str,
        required=True,
        help='Directory containing checkpoint files',
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='epoch_visualizations',
        help='Output directory for images',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        nargs='+',
        default=None,
        help='Specific epochs to visualize (default: all)',
    )
    parser.add_argument(
        '--globe_only',
        action='store_true',
        help='Only generate globe visualizations',
    )
    parser.add_argument(
        '--splines_only',
        action='store_true',
        help='Only generate spline visualizations',
    )
    parser.add_argument(
        '--no_combined',
        action='store_true',
        help='Skip combined dashboard visualization',
    )
    parser.add_argument(
        '--lat_resolution',
        type=int,
        default=90,
        help='Latitude resolution for globe grid',
    )
    parser.add_argument(
        '--lon_resolution',
        type=int,
        default=180,
        help='Longitude resolution for globe grid',
    )

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find checkpoint files
    patterns = ['*.ckpt', '*.pt', 'epoch_*.pt']
    checkpoint_files = []
    for pattern in patterns:
        checkpoint_files.extend(checkpoint_dir.glob(pattern))

    checkpoint_files = sorted(set(checkpoint_files))

    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return

    print(f"Found {len(checkpoint_files)} checkpoints")

    # Filter by epoch if specified
    if args.epochs is not None:
        epoch_set = set(args.epochs)
        checkpoint_files = [
            f for f in checkpoint_files
            if get_epoch_from_filename(f.name) in epoch_set
        ]
        print(f"Filtering to {len(checkpoint_files)} specified epochs")

    # Sort by epoch
    checkpoint_files = sorted(checkpoint_files, key=lambda f: get_epoch_from_filename(f.name))

    # Create globe grid
    lats, lons, coords = create_globe_grid(args.lat_resolution, args.lon_resolution)
    coords_tensor = torch.tensor(coords, dtype=torch.float32)

    # Process each checkpoint
    for ckpt_path in checkpoint_files:
        process_checkpoint(
            ckpt_path,
            output_dir,
            lats,
            lons,
            coords_tensor,
            args.lat_resolution,
            args.lon_resolution,
            globe_only=args.globe_only,
            splines_only=args.splines_only,
            combined=not args.no_combined,
        )

    print(f"\nDone! Generated visualizations in {output_dir}")

    # Provide ffmpeg tip
    print("\nTip: Create animation with ffmpeg:")
    print(f"  ffmpeg -framerate 4 -pattern_type glob -i '{output_dir}/combined_epoch_*.png' -c:v libx264 -pix_fmt yuv420p training_evolution.mp4")


if __name__ == '__main__':
    main()
