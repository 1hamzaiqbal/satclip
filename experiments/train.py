#!/usr/bin/env python
"""Main training script for learned activations experiments.

Usage:
    # Using config files
    python -m experiments.train --config experiments/configs/experiments/elevation.yaml

    # Composing configs
    python -m experiments.train \\
        --config experiments/configs/base.yaml \\
        --config experiments/configs/experiments/elevation.yaml \\
        --config experiments/configs/encodings/sh_l10.yaml \\
        --config experiments/configs/activations/spline.yaml

    # With CLI overrides
    python -m experiments.train \\
        --config experiments/configs/experiments/elevation.yaml \\
        --model.hidden_dim=512 \\
        --training.learning_rate=0.0001

    # Quick test run
    python -m experiments.train --config ... --trainer.fast_dev_run=true
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils.config import (
    load_config,
    merge_configs,
    parse_cli_overrides,
    Config,
)
from experiments.models.lightning_module import LearnedActivationsModule, ContrastiveLearningModule
from experiments.data import (
    GeographicDataModule,
    SyntheticDataModule,
    SatCLIPHuggingFaceDataModule,
    HFMultispectralDataModule,
)
from experiments.callbacks import SplineVisualizationCallback, GlobeVisualizationCallback, EpochLoggerCallback
from experiments.utils.paths import get_paths, Paths


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train learned activations models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        action="append",
        default=[],
        help="Config file(s) to load. Can specify multiple for composition.",
    )

    # Allow arbitrary --key=value overrides
    args, unknown = parser.parse_known_args()

    # Parse CLI overrides
    cli_overrides = parse_cli_overrides(unknown)

    return args.config, cli_overrides


def get_datamodule(config: Config) -> pl.LightningDataModule:
    """Create datamodule from config.

    Args:
        config: Configuration object

    Returns:
        Lightning DataModule
    """
    data_config = config.data
    dataset = data_config.get("dataset", "checkerboard")

    if dataset in ["checkerboard", "gaussian_mixture", "sinusoidal"]:
        return SyntheticDataModule(
            dataset_type=dataset,
            n_samples=data_config.get("n_samples", 10000),
            test_fraction=data_config.get("test_fraction", 0.3),
            batch_size=data_config.get("batch_size", 256),
            num_workers=data_config.get("num_workers", 4),
            seed=config.experiment.get("seed", 42),
            grid_size=data_config.get("grid_size", 10.0),
            noise=data_config.get("noise", 0.0),
            region=data_config.get("region", None),
        )

    elif dataset in ["elevation", "population"]:
        return GeographicDataModule(
            dataset_task=dataset,
            n_samples=data_config.get("n_samples", 10000),
            region=data_config.get("region", None),
            test_fraction=data_config.get("test_fraction", 0.3),
            batch_size=data_config.get("batch_size", 256),
            num_workers=data_config.get("num_workers", 4),
            seed=config.experiment.get("seed", 42),
            data_path=data_config.get("data_path", None),
        )

    elif dataset == "satclip":
        return SatCLIPHuggingFaceDataModule(
            batch_size=data_config.get("batch_size", 512),
            num_workers=data_config.get("num_workers", 8),
            val_split=data_config.get("val_split", 0.1),
            cache_dir=data_config.get("cache_dir", None),
        )

    elif dataset == "satclip_multispectral" or data_config.get("use_hf_dataset", False):
        # HuggingFace Arrow-based multispectral dataset (fast, preprocessed)
        # Use paths module for default dataset path if not specified
        dataset_path = data_config.get("hf_dataset_path")
        if not dataset_path:
            dataset_path = Paths.get_dataset_path()
        vision_encoder = config.model.get("vision_encoder", "")
        crop_size = data_config.get("crop_size", None)
        if vision_encoder in ["moco_vit16"]:
            if crop_size != 224:
                print(f"Adjusting crop_size to 224 for vision encoder {vision_encoder}")
            crop_size = 224
        if crop_size is None:
            crop_size = 256
        return HFMultispectralDataModule(
            dataset_path=dataset_path,
            batch_size=data_config.get("batch_size", 256),
            num_workers=data_config.get("num_workers", 8),
            crop_size=crop_size,
            val_split=data_config.get("val_split", 0.1),
            pad_to_13_channels=data_config.get("pad_to_13_channels", True),
            preprocessed=data_config.get("preprocessed", False),
            seed=config.experiment.get("seed", 42),
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_model(config: Config, datamodule: pl.LightningDataModule) -> pl.LightningModule:
    """Create model from config.

    Args:
        config: Configuration object
        datamodule: DataModule (to get target statistics)

    Returns:
        Lightning Module
    """
    model_config = config.model
    training_config = config.training
    task = config.data.get("task", "regression")

    # Check if this is a contrastive learning task
    if task == "contrastive":
        lr_location = training_config.get("lr_location", None)
        lr_image_proj = training_config.get("lr_image_proj", None)
        lr_logit_scale = training_config.get("lr_logit_scale", None)
        lr_backbone = training_config.get("lr_backbone", None)
        gather_negatives = training_config.get("gather_negatives", False)
        activation_freeze_epoch = training_config.get("activation_freeze_epoch", None)
        return ContrastiveLearningModule(
            # Vision encoder
            vision_encoder=model_config.get("vision_encoder", "moco_resnet18"),
            embed_dim=model_config.get("embed_dim", 256),
            freeze_vision=model_config.get("freeze_vision", True),
            # Location encoder - encoding
            encoding_type=model_config.encoding.get("type", "spherical_harmonics"),
            encoding_config={
                k: v for k, v in model_config.encoding.to_dict().items() if k != "type"
            },
            # Location encoder - network
            hidden_dim=model_config.network.get("hidden_dim", 256),
            num_layers=model_config.network.get("num_layers", 2),
            # Location encoder - activation
            activation_type=model_config.activation.get("type", "siren"),
            activation_config={
                k: v for k, v in model_config.activation.to_dict().items() if k != "type"
            },
            # Training
            learning_rate=training_config.get("learning_rate", 1e-4),
            weight_decay=training_config.get("weight_decay", 0.01),
            temperature=model_config.get("temperature", 0.07),
            lr_location=lr_location,
            lr_image_proj=lr_image_proj,
            lr_logit_scale=lr_logit_scale,
            lr_backbone=lr_backbone,
            gather_negatives=gather_negatives,
            activation_freeze_epoch=activation_freeze_epoch,
            # Scheduler
            scheduler=training_config.get("scheduler", None),
            warmup_epochs=training_config.get("warmup_epochs", 10),
            min_lr=training_config.get("min_lr", 1e-6),
        )

    # Standard regression/classification task
    # Get target statistics if available
    target_mean = 0.0
    target_std = 1.0
    if hasattr(datamodule, "target_mean"):
        target_mean = float(datamodule.target_mean)
        target_std = float(datamodule.target_std)

    return LearnedActivationsModule(
        # Encoding
        encoding_type=model_config.encoding.get("type", "spherical_harmonics"),
        encoding_config={
            k: v for k, v in model_config.encoding.to_dict().items() if k != "type"
        },
        # Network
        hidden_dim=model_config.network.get("hidden_dim", 256),
        num_layers=model_config.network.get("num_layers", 3),
        # Activation
        activation_type=model_config.activation.get("type", "relu"),
        activation_config={
            k: v for k, v in model_config.activation.to_dict().items() if k != "type"
        },
        # Task
        task=task,
        num_classes=config.data.get("num_classes", 1),
        # Training
        learning_rate=training_config.get("learning_rate", 0.001),
        weight_decay=training_config.get("weight_decay", 0.0),
        # Normalization
        target_mean=target_mean,
        target_std=target_std,
    )


def get_callbacks(config: Config, checkpoint_dir: str = None) -> List[pl.Callback]:
    """Create callbacks from config.

    Args:
        config: Configuration object
        checkpoint_dir: Directory to save checkpoints (if None, uses logger default)

    Returns:
        List of callbacks
    """
    callbacks = []
    logging_config = config.logging
    training_config = config.training

    # Model checkpoint
    checkpoint_config = logging_config.get("checkpoint", Config({}))
    callbacks.append(
        ModelCheckpoint(
            dirpath=checkpoint_dir,  # Explicit directory for checkpoints
            monitor=checkpoint_config.get("monitor", "val_loss"),
            mode=checkpoint_config.get("mode", "min"),
            save_top_k=checkpoint_config.get("save_top_k", 1),
            save_last=checkpoint_config.get("save_last", True),
            filename="{epoch}-{val_loss:.4f}",
            every_n_epochs=checkpoint_config.get("every_n_epochs", None),
        )
    )

    # Early stopping
    early_stopping = training_config.get("early_stopping", Config({}))
    if early_stopping.get("enabled", False):
        callbacks.append(
            EarlyStopping(
                monitor=early_stopping.get("monitor", "val_loss"),
                mode=early_stopping.get("mode", "min"),
                patience=early_stopping.get("patience", 10),
            )
        )

    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # Epoch logger for clear progress in SLURM logs
    callbacks.append(EpochLoggerCallback())

    # Spline visualization callback (only if using spline activation)
    activation_type = config.model.activation.get("type", "relu")
    if activation_type == "spline":
        spline_viz_config = logging_config.get("spline_viz", Config({}))
        callbacks.append(
            SplineVisualizationCallback(
                log_every_n_epochs=spline_viz_config.get("log_every_n_epochs", 10),
                log_on_train_start=spline_viz_config.get("log_on_train_start", True),
                input_range=tuple(spline_viz_config.get("input_range", [-4.0, 4.0])),
            )
        )

    # Globe visualization callback (for contrastive learning)
    task = config.data.get("task", "regression")
    globe_viz_config = logging_config.get("globe_viz", Config({}))
    if task == "contrastive" and globe_viz_config.get("enabled", True):
        callbacks.append(
            GlobeVisualizationCallback(
                log_every_n_epochs=globe_viz_config.get("log_every_n_epochs", 10),
                log_on_train_start=globe_viz_config.get("log_on_train_start", True),
                lat_resolution=globe_viz_config.get("lat_resolution", 60),
                lon_resolution=globe_viz_config.get("lon_resolution", 120),
            )
        )

    return callbacks


def get_logger(config: Config) -> pl.loggers.Logger:
    """Create logger from config.

    Args:
        config: Configuration object

    Returns:
        Logger instance
    """
    logging_config = config.logging
    logger_type = logging_config.get("logger", "tensorboard")

    experiment_name = config.experiment.get("name", "experiment")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Use paths module for default save_dir if not specified
    save_dir = logging_config.get("save_dir")
    if not save_dir:
        save_dir = Paths.get_logs_dir()

    if logger_type == "tensorboard":
        return TensorBoardLogger(
            save_dir=save_dir,
            name=logging_config.get("name", "experiments"),
            version=f"{experiment_name}_{timestamp}",
        )
    elif logger_type == "csv":
        return CSVLogger(
            save_dir=save_dir,
            name=f"{experiment_name}_{timestamp}",
        )
    else:
        # Default to TensorBoard
        return TensorBoardLogger(
            save_dir=save_dir,
            name=experiment_name,
        )


def main():
    """Main training function."""
    # Parse arguments
    config_paths, cli_overrides = parse_args()

    # Load base config
    base_config_path = Path(__file__).parent / "configs" / "base.yaml"
    configs = [load_config(base_config_path)]

    # Load specified configs
    for path in config_paths:
        configs.append(load_config(path))

    # Merge all configs
    merged = merge_configs(configs)

    # Apply CLI overrides
    merged = merge_configs([merged, cli_overrides])

    # Convert to Config object
    config = Config(merged)

    # Print detailed config
    print("=" * 60)
    print("Configuration Summary")
    print("=" * 60)
    print(f"Experiment: {config.experiment.get('name', 'unknown')}")
    print(f"Encoding: {config.model.encoding.get('type', 'unknown')}")
    print(f"Activation: {config.model.activation.get('type', 'unknown')}")
    print(f"Dataset: {config.data.get('dataset', 'unknown')}")
    print(f"Task: {config.data.get('task', 'unknown')}")
    print("-" * 60)
    print("Training:")
    print(f"  Max epochs: {config.training.get('max_epochs', 'N/A')}")
    print(f"  Learning rate: {config.training.get('learning_rate', 'N/A')}")
    print(f"  Batch size: {config.data.get('batch_size', 'N/A')}")
    print(f"  Accumulate grad batches: {config.training.get('accumulate_grad_batches', 1)}")
    print("-" * 60)
    print("Logging:")
    print(f"  Save dir: {config.logging.get('save_dir', './logs')}")
    print(f"  Logger: {config.logging.get('logger', 'tensorboard')}")
    print(f"  Log every N steps: {config.logging.get('log_every_n_steps', 50)}")
    checkpoint_cfg = config.logging.get("checkpoint", Config({}))
    print(f"  Checkpoint every N epochs: {checkpoint_cfg.get('every_n_epochs', 'end only')}")
    print(f"  Save top K: {checkpoint_cfg.get('save_top_k', 1)}")
    print("=" * 60)
    sys.stdout.flush()  # Ensure output is written immediately

    # Set seed
    seed = config.experiment.get("seed", 42)
    pl.seed_everything(seed)

    # Create datamodule
    print("\nSetting up data...")
    datamodule = get_datamodule(config)
    datamodule.setup()

    # Create model
    print("Creating model...")
    model = get_model(config, datamodule)

    # Create logger first (to get checkpoint directory)
    logger = get_logger(config)

    # Compute checkpoint directory based on logger's log_dir
    checkpoint_dir = os.path.join(logger.log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Create callbacks with explicit checkpoint directory
    callbacks = get_callbacks(config, checkpoint_dir=checkpoint_dir)
    print(f"\nCallbacks ({len(callbacks)}):")
    for cb in callbacks:
        print(f"  - {cb.__class__.__name__}")
    sys.stdout.flush()

    # Create trainer
    hardware = config.get("hardware", Config({}))
    training = config.training

    logging_config = config.logging

    trainer = pl.Trainer(
        max_epochs=training.get("max_epochs", 100),
        accelerator=hardware.get("accelerator", "auto"),
        devices=hardware.get("devices", 1),
        precision=hardware.get("precision", 32),
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=training.get("gradient_clip_val", None),
        accumulate_grad_batches=training.get("accumulate_grad_batches", 1),
        log_every_n_steps=logging_config.get("log_every_n_steps", 50),
        enable_progress_bar=True,
        deterministic=config.get("reproducibility", Config({})).get("deterministic", True),
    )

    # Print trainer info
    print("\nTrainer configuration:")
    print(f"  Accelerator: {trainer.accelerator}")
    print(f"  Devices: {trainer.device_ids}")
    print(f"  Precision: {trainer.precision}")
    print(f"  Max epochs: {trainer.max_epochs}")
    print(f"  Log dir: {logger.log_dir}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    sys.stdout.flush()

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    sys.stdout.flush()

    trainer.fit(model, datamodule=datamodule)

    # Print results
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)

    if trainer.checkpoint_callback.best_model_path:
        print(f"Best model path: {trainer.checkpoint_callback.best_model_path}")
        print(f"Best val_loss: {trainer.checkpoint_callback.best_model_score:.4f}")
    else:
        print("No best model checkpoint found (training may have failed early)")

    # List all saved checkpoints
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        print(f"\nSaved checkpoints ({len(checkpoints)}):")
        for ckpt in sorted(checkpoints):
            print(f"  - {ckpt}")

    sys.stdout.flush()


if __name__ == "__main__":
    main()
