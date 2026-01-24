#!/usr/bin/env python
"""Visualize spline activation shapes from a trained checkpoint.

Usage:
    # From a checkpoint:
    python -m experiments.scripts.analysis.visualize_splines \
        --checkpoint /path/to/checkpoint.ckpt \
        --output splines.png

    # Compare multiple checkpoints:
    python -m experiments.scripts.analysis.visualize_splines \
        --checkpoint ckpt1.ckpt ckpt2.ckpt ckpt3.ckpt \
        --labels "Epoch 10" "Epoch 50" "Epoch 100" \
        --output spline_evolution.png

    # Save individual layer plots:
    python -m experiments.scripts.analysis.visualize_splines \
        --checkpoint model.ckpt \
        --output-dir ./spline_plots/
"""

import argparse
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.models.lightning_module import ContrastiveLearningModule
from experiments.models.activations.spline import SplineActivation


def find_spline_activations(model) -> List[tuple]:
    """Find all SplineActivation modules in the model."""
    splines = []
    for name, module in model.named_modules():
        if isinstance(module, SplineActivation):
            splines.append((name, module))
    return splines


def plot_spline(
    spline: SplineActivation,
    ax: plt.Axes,
    label: str = None,
    color: str = None,
    input_range: tuple = (-4.0, 4.0),
    n_points: int = 200,
    show_knots: bool = True,
    show_reference: bool = True,
):
    """Plot a single spline activation on the given axes."""
    # Get knot data
    knot_x, knot_y = spline.get_knot_data()

    # Evaluate spline over range
    x = torch.linspace(input_range[0], input_range[1], n_points)
    with torch.no_grad():
        y = spline(x.to(knot_x.device)).cpu()

    x_np = x.numpy()
    y_np = y.numpy()

    # Plot the spline curve
    line_kwargs = {'linewidth': 2}
    if label:
        line_kwargs['label'] = label
    if color:
        line_kwargs['color'] = color

    ax.plot(x_np, y_np, **line_kwargs)

    # Plot the knot points
    if show_knots:
        scatter_kwargs = {'s': 30, 'zorder': 5, 'alpha': 0.7}
        if color:
            scatter_kwargs['c'] = color
        else:
            scatter_kwargs['c'] = 'red'
        ax.scatter(knot_x.numpy(), knot_y.numpy(), **scatter_kwargs)

    # Plot reference functions
    if show_reference:
        ax.plot(x_np, np.maximum(0, x_np), 'k--', alpha=0.3, linewidth=1, label='ReLU' if label is None else None)
        ax.plot(x_np, x_np, 'k:', alpha=0.2, linewidth=1)

    return knot_x, knot_y


def visualize_checkpoint(
    checkpoint_path: str,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    input_range: tuple = (-4.0, 4.0),
    figsize: tuple = (10, 6),
):
    """Visualize splines from a single checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load model
    model = ContrastiveLearningModule.load_from_checkpoint(
        checkpoint_path,
        map_location='cpu',
    )
    model.eval()

    # Find spline activations
    splines = find_spline_activations(model)

    if not splines:
        print("No SplineActivation modules found in the model.")
        return

    print(f"Found {len(splines)} spline activation(s)")

    # Create combined figure
    n_splines = len(splines)
    n_cols = min(3, n_splines)
    n_rows = (n_splines + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_splines == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (name, spline) in enumerate(splines):
        ax = axes[idx]
        plot_spline(spline, ax, input_range=input_range)

        # Get layer info
        short_name = name.split('.')[-2] if '.' in name else name
        ax.set_title(f'Layer: {short_name}')
        ax.set_xlabel('Input')
        ax.set_ylabel('Output')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(input_range)

        # Add info text
        info_text = f"k={spline.n_knots}, range={spline.input_range}"
        ax.text(
            0.02, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    # Hide empty subplots
    for idx in range(n_splines, len(axes)):
        axes[idx].set_visible(False)

    checkpoint_name = Path(checkpoint_path).stem
    fig.suptitle(f'Spline Activations: {checkpoint_name}', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved combined figure to: {output_path}")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save individual plots
        for name, spline in splines:
            fig_single, ax_single = plt.subplots(1, 1, figsize=figsize)
            plot_spline(spline, ax_single, input_range=input_range)

            clean_name = name.replace('.', '_')
            ax_single.set_title(f'Spline: {name}')
            ax_single.set_xlabel('Input')
            ax_single.set_ylabel('Output')
            ax_single.grid(True, alpha=0.3)
            ax_single.legend()

            single_path = output_dir / f'{clean_name}.png'
            plt.savefig(single_path, dpi=150, bbox_inches='tight')
            plt.close(fig_single)
            print(f"Saved: {single_path}")

    plt.show()


def compare_checkpoints(
    checkpoint_paths: List[str],
    labels: List[str],
    output_path: str,
    input_range: tuple = (-4.0, 4.0),
    figsize: tuple = (12, 8),
):
    """Compare splines from multiple checkpoints (e.g., different epochs)."""
    print(f"Comparing {len(checkpoint_paths)} checkpoints...")

    colors = plt.cm.viridis(np.linspace(0, 1, len(checkpoint_paths)))

    # Load all models
    models = []
    for ckpt_path in checkpoint_paths:
        model = ContrastiveLearningModule.load_from_checkpoint(
            ckpt_path,
            map_location='cpu',
        )
        model.eval()
        models.append(model)

    # Get spline names from first model
    splines_first = find_spline_activations(models[0])
    if not splines_first:
        print("No SplineActivation modules found.")
        return

    n_splines = len(splines_first)
    n_cols = min(3, n_splines)
    n_rows = (n_splines + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_splines == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (name, _) in enumerate(splines_first):
        ax = axes[idx]

        # Plot each checkpoint's spline
        for model, label, color in zip(models, labels, colors):
            splines = find_spline_activations(model)
            for spline_name, spline in splines:
                if spline_name == name:
                    plot_spline(
                        spline, ax,
                        label=label,
                        color=color,
                        input_range=input_range,
                        show_knots=False,
                        show_reference=(label == labels[0]),
                    )
                    break

        short_name = name.split('.')[-2] if '.' in name else name
        ax.set_title(f'Layer: {short_name}')
        ax.set_xlabel('Input')
        ax.set_ylabel('Output')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(input_range)
        ax.legend(fontsize=8)

    # Hide empty subplots
    for idx in range(n_splines, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Spline Evolution Across Training', fontsize=14)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison figure to: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize spline activations from trained checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        nargs='+',
        required=True,
        help='Checkpoint file(s) to visualize',
    )
    parser.add_argument(
        '--labels', '-l',
        type=str,
        nargs='+',
        default=None,
        help='Labels for multiple checkpoints (for comparison)',
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path for combined figure',
    )
    parser.add_argument(
        '--output-dir', '-d',
        type=str,
        default=None,
        help='Output directory for individual layer plots',
    )
    parser.add_argument(
        '--input-range',
        type=float,
        nargs=2,
        default=[-4.0, 4.0],
        help='Input range for plotting (default: -4 4)',
    )

    args = parser.parse_args()

    input_range = tuple(args.input_range)

    if len(args.checkpoint) == 1:
        # Single checkpoint visualization
        visualize_checkpoint(
            args.checkpoint[0],
            output_path=args.output,
            output_dir=args.output_dir,
            input_range=input_range,
        )
    else:
        # Multiple checkpoint comparison
        labels = args.labels or [f'Checkpoint {i+1}' for i in range(len(args.checkpoint))]
        if len(labels) != len(args.checkpoint):
            raise ValueError("Number of labels must match number of checkpoints")

        output_path = args.output or 'spline_comparison.png'
        compare_checkpoints(
            args.checkpoint,
            labels,
            output_path,
            input_range=input_range,
        )


if __name__ == '__main__':
    main()
