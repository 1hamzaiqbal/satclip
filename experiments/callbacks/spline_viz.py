"""Spline visualization callback for TensorBoard logging.

Logs spline activation shapes during training to monitor how the
learned activation functions evolve.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from experiments.models.activations.spline import SplineActivation


class SplineVisualizationCallback(Callback):
    """Callback to visualize spline activation shapes during training.

    Finds all SplineActivation modules in the model and logs their
    shapes to TensorBoard at specified intervals.
    """

    def __init__(
        self,
        log_every_n_epochs: int = 10,
        log_on_train_start: bool = True,
        input_range: tuple = (-4.0, 4.0),
        n_points: int = 200,
    ):
        """Initialize callback.

        Args:
            log_every_n_epochs: Log spline shapes every N epochs
            log_on_train_start: Whether to log initial spline shapes
            input_range: Range for plotting (can extend beyond spline range)
            n_points: Number of points for smooth curve
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.log_on_train_start = log_on_train_start
        self.input_range = input_range
        self.n_points = n_points

    def _find_spline_activations(self, model: pl.LightningModule) -> List[tuple]:
        """Find all SplineActivation modules in the model.

        Returns:
            List of (name, module) tuples
        """
        splines = []
        for name, module in model.named_modules():
            if isinstance(module, SplineActivation):
                splines.append((name, module))
        return splines

    def _create_spline_figure(
        self,
        spline: SplineActivation,
        name: str,
        epoch: int,
    ) -> plt.Figure:
        """Create a figure showing the spline shape.

        Args:
            spline: SplineActivation module
            name: Name of the module
            epoch: Current epoch

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Get knot data (returns CPU tensors)
        knot_x, knot_y = spline.get_knot_data()

        # Get device from spline's actual parameters (may be on GPU)
        spline_device = next(spline.parameters()).device

        # Evaluate spline over range
        x = torch.linspace(self.input_range[0], self.input_range[1], self.n_points)
        with torch.no_grad():
            y = spline(x.to(spline_device)).cpu()

        # Plot the spline curve
        ax.plot(x.numpy(), y.numpy(), 'b-', linewidth=2, label='Learned spline')

        # Plot the knot points
        ax.scatter(
            knot_x.numpy(), knot_y.numpy(),
            c='red', s=50, zorder=5, label='Knots'
        )

        # Plot reference functions for comparison
        x_np = x.numpy()
        ax.plot(x_np, np.maximum(0, x_np), 'g--', alpha=0.5, linewidth=1, label='ReLU')
        ax.plot(x_np, x_np, 'k:', alpha=0.3, linewidth=1, label='Identity')

        # Formatting
        ax.set_xlabel('Input', fontsize=12)
        ax.set_ylabel('Output', fontsize=12)
        ax.set_title(f'Spline Activation: {name}\nEpoch {epoch}', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(self.input_range)

        # Add info text
        info_text = (
            f"n_knots: {spline.n_knots}\n"
            f"range: {spline.input_range}\n"
            f"init: {spline.init_type}"
        )
        ax.text(
            0.98, 0.02, info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        plt.tight_layout()
        return fig

    def _create_combined_figure(
        self,
        splines: List[tuple],
        epoch: int,
    ) -> plt.Figure:
        """Create a combined figure showing all splines.

        Args:
            splines: List of (name, module) tuples
            epoch: Current epoch

        Returns:
            Matplotlib figure
        """
        n_splines = len(splines)
        if n_splines == 0:
            return None

        # Create figure with subplots
        n_cols = min(3, n_splines)
        n_rows = (n_splines + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))

        if n_splines == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        x = torch.linspace(self.input_range[0], self.input_range[1], self.n_points)
        x_np = x.numpy()

        for idx, (name, spline) in enumerate(splines):
            ax = axes[idx]

            # Get knot data (returns CPU tensors)
            knot_x, knot_y = spline.get_knot_data()

            # Get device from spline's actual parameters (may be on GPU)
            spline_device = next(spline.parameters()).device

            # Evaluate spline
            with torch.no_grad():
                y = spline(x.to(spline_device)).cpu()

            # Plot
            ax.plot(x_np, y.numpy(), 'b-', linewidth=2, label='Learned')
            ax.scatter(knot_x.numpy(), knot_y.numpy(), c='red', s=30, zorder=5)
            ax.plot(x_np, np.maximum(0, x_np), 'g--', alpha=0.4, linewidth=1, label='ReLU')

            # Get layer number from name
            short_name = name.split('.')[-2] if '.' in name else name
            ax.set_title(f'Layer: {short_name}', fontsize=10)
            ax.set_xlabel('Input')
            ax.set_ylabel('Output')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(self.input_range)

        # Hide empty subplots
        for idx in range(n_splines, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(f'Spline Activations - Epoch {epoch}', fontsize=14)
        plt.tight_layout()
        return fig

    def _log_splines(self, trainer: pl.Trainer, pl_module: pl.LightningModule, epoch: int):
        """Log spline visualizations to TensorBoard.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
            epoch: Current epoch
        """
        splines = self._find_spline_activations(pl_module)

        if not splines:
            return

        # Log combined figure
        combined_fig = self._create_combined_figure(splines, epoch)
        if combined_fig is not None and trainer.logger is not None:
            trainer.logger.experiment.add_figure(
                'splines/all_layers',
                combined_fig,
                global_step=epoch,
            )
            plt.close(combined_fig)

        # Log individual splines
        for name, spline in splines:
            fig = self._create_spline_figure(spline, name, epoch)
            if trainer.logger is not None:
                # Clean up name for TensorBoard
                clean_name = name.replace('.', '/')
                trainer.logger.experiment.add_figure(
                    f'splines/{clean_name}',
                    fig,
                    global_step=epoch,
                )
            plt.close(fig)

        # Also log knot values as scalars for tracking evolution
        for name, spline in splines:
            knot_x, knot_y = spline.get_knot_data()
            clean_name = name.replace('.', '_')

            # Log min, max, mean of knot values
            if trainer.logger is not None:
                trainer.logger.experiment.add_scalar(
                    f'spline_knots/{clean_name}/mean',
                    knot_y.mean().item(),
                    global_step=epoch,
                )
                trainer.logger.experiment.add_scalar(
                    f'spline_knots/{clean_name}/std',
                    knot_y.std().item(),
                    global_step=epoch,
                )

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Log initial spline shapes."""
        if self.log_on_train_start:
            self._log_splines(trainer, pl_module, epoch=0)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Log spline shapes at end of epoch."""
        current_epoch = trainer.current_epoch

        # Log every N epochs, starting from epoch 1 (0 is logged at start)
        if (current_epoch + 1) % self.log_every_n_epochs == 0:
            self._log_splines(trainer, pl_module, epoch=current_epoch + 1)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Log final spline shapes."""
        current_epoch = trainer.current_epoch
        # Only log if not already logged this epoch
        if (current_epoch + 1) % self.log_every_n_epochs != 0:
            self._log_splines(trainer, pl_module, epoch=current_epoch + 1)
