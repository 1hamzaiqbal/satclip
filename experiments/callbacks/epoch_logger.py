"""Simple epoch logging callback for clear progress tracking in SLURM logs."""

import sys
import time
from typing import Optional
import lightning.pytorch as pl


class EpochLoggerCallback(pl.Callback):
    """Prints epoch progress with loss values.

    This callback provides clear, readable progress logs that work well
    in SLURM output files (unlike progress bars that use carriage returns).

    Output format:
        Epoch 1/50 | Train Loss: 3.2145 | Val Loss: 3.1892 | Time: 2m 34s
    """

    def __init__(self):
        super().__init__()
        self.epoch_start_time: Optional[float] = None
        self.train_loss: Optional[float] = None

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Record epoch start time."""
        self.epoch_start_time = time.time()
        self.train_loss = None

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Store training loss for later printing."""
        # Get train loss from logged metrics
        if "train_loss" in trainer.callback_metrics:
            self.train_loss = float(trainer.callback_metrics["train_loss"])

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Print epoch summary after validation completes."""
        # Skip sanity check
        if trainer.sanity_checking:
            return

        # Calculate elapsed time
        elapsed = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        time_str = f"{minutes}m {seconds:02d}s"

        # Get metrics
        current_epoch = trainer.current_epoch
        max_epochs = trainer.max_epochs

        val_loss = trainer.callback_metrics.get("val_loss", None)
        train_loss = self.train_loss

        # Build output string
        parts = [f"Epoch {current_epoch + 1}/{max_epochs}"]

        if train_loss is not None:
            parts.append(f"Train Loss: {train_loss:.4f}")
        if val_loss is not None:
            parts.append(f"Val Loss: {float(val_loss):.4f}")

        parts.append(f"Time: {time_str}")

        # Print and flush
        print(" | ".join(parts))
        sys.stdout.flush()
