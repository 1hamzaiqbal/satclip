"""PyTorch Lightning modules for training location encoders.

Provides:
1. LearnedActivationsModule - For regression/classification tasks
2. ContrastiveLearningModule - For SatCLIP-style contrastive learning with modular activations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from typing import Dict, Any, Optional, Literal, Tuple
import numpy as np

try:
    import timm
    from torchgeo.models import ResNet18_Weights, ResNet50_Weights, ViTSmall16_Weights
    HAS_TORCHGEO = True
except ImportError:
    HAS_TORCHGEO = False

from .location_encoder import LocationEncoder, LocationRegressor


class LearnedActivationsModule(pl.LightningModule):
    """Lightning module for coordinate-based regression/classification.

    Supports:
    - Regression tasks (MSE loss)
    - Classification tasks (CrossEntropy loss)
    - Various encoding/activation combinations
    """

    def __init__(
        self,
        # Encoding config
        encoding_type: str = "spherical_harmonics",
        encoding_config: Optional[Dict] = None,
        # Network config
        hidden_dim: int = 256,
        num_layers: int = 3,
        # Activation config
        activation_type: str = "relu",
        activation_config: Optional[Dict] = None,
        # Task config
        task: Literal["regression", "classification"] = "regression",
        num_classes: int = 1,
        # Training config
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        # Normalization
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ):
        """Initialize module.

        Args:
            encoding_type: Positional encoding type
            encoding_config: Encoding configuration
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            activation_type: Activation function type
            activation_config: Activation configuration
            task: "regression" or "classification"
            num_classes: Number of classes (for classification) or output dim
            learning_rate: Learning rate
            weight_decay: Weight decay for AdamW
            target_mean: Mean for target normalization (regression)
            target_std: Std for target normalization (regression)
        """
        super().__init__()
        self.save_hyperparameters()

        self.task = task
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.target_mean = target_mean
        self.target_std = target_std

        # Build model
        encoding_config = encoding_config or {}
        activation_config = activation_config or {}

        self.model = LocationRegressor(
            encoding_type=encoding_type,
            encoding_config=encoding_config,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation_type=activation_type,
            activation_config=activation_config,
            output_dim=num_classes,
        )

        # Loss function
        if task == "regression":
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            coords: (batch, 2) coordinates

        Returns:
            Predictions
        """
        return self.model(coords)

    def _normalize_target(self, y: torch.Tensor) -> torch.Tensor:
        """Normalize target for training."""
        return (y - self.target_mean) / self.target_std

    def _denormalize_pred(self, pred: torch.Tensor) -> torch.Tensor:
        """Denormalize predictions for evaluation."""
        return pred * self.target_std + self.target_mean

    def training_step(self, batch, batch_idx):
        """Training step."""
        coords, targets = batch["coords"], batch["target"]

        preds = self(coords)

        if self.task == "regression":
            targets_norm = self._normalize_target(targets)
            loss = self.loss_fn(preds, targets_norm)
        else:
            loss = self.loss_fn(preds, targets.long())

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step with R² computation."""
        coords, targets = batch["coords"], batch["target"]

        preds = self(coords)

        if self.task == "regression":
            targets_norm = self._normalize_target(targets)
            loss = self.loss_fn(preds, targets_norm)

            # Denormalize for R² calculation
            preds_denorm = self._denormalize_pred(preds)
            r2 = self._compute_r2(targets, preds_denorm)

            self.log("val_loss", loss, prog_bar=True)
            self.log("val_r2", r2, prog_bar=True)
        else:
            loss = self.loss_fn(preds, targets.long())
            acc = (preds.argmax(dim=-1) == targets).float().mean()

            self.log("val_loss", loss, prog_bar=True)
            self.log("val_acc", acc, prog_bar=True)

        return loss

    def _compute_r2(
        self, y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> torch.Tensor:
        """Compute R² score."""
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        return 1 - (ss_res / ss_tot)

    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer


class ContrastiveLearningModule(pl.LightningModule):
    """Lightning module for SatCLIP-style contrastive learning.

    Learns location embeddings that align with image embeddings
    using a contrastive loss. Supports modular activation functions
    (SIREN, splines, RFF, ReLU) for the location encoder.

    This is the key module for training your own SatCLIP with custom
    learned activation functions.
    """

    def __init__(
        self,
        # Vision encoder config
        vision_encoder: str = "moco_resnet18",
        embed_dim: int = 256,
        freeze_vision: bool = True,
        # Location encoder config
        encoding_type: str = "spherical_harmonics",
        encoding_config: Optional[Dict] = None,
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation_type: str = "siren",
        activation_config: Optional[Dict] = None,
        # Training config
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        temperature: float = 0.07,
        # Scheduler config
        scheduler: Optional[str] = None,
        warmup_epochs: int = 10,
        min_lr: float = 1e-6,
    ):
        """Initialize contrastive learning module.

        Args:
            vision_encoder: Vision encoder type. Options:
                - "moco_resnet18": MoCo pretrained ResNet18 (recommended, fast)
                - "moco_resnet50": MoCo pretrained ResNet50 (more capacity)
                - "moco_vit16": MoCo pretrained ViT-Small-16 (best quality)
            embed_dim: Embedding dimension (256 is standard for SatCLIP)
            freeze_vision: Whether to freeze vision encoder (except final layer)
            encoding_type: Location encoding type. Options:
                - "spherical_harmonics" or "sh": Spherical harmonics (recommended)
                - "raw": Raw normalized coordinates
                - "rff": Random Fourier Features encoding
                - "cartesian3d": 3D Cartesian projection
            encoding_config: Location encoding config dict
            hidden_dim: Location encoder hidden dim
            num_layers: Location encoder layers
            activation_type: Location encoder activation. Options:
                - "siren": SIREN (sinusoidal, original SatCLIP)
                - "relu": ReLU (often better than SIREN!)
                - "spline": Learnable spline activation
                - "rff": RFF activation (WARNING: avoid with SH encoding!)
            activation_config: Activation config dict
            learning_rate: Learning rate
            weight_decay: Weight decay
            temperature: Contrastive loss temperature
            scheduler: Learning rate scheduler ("cosine", "warmup_cosine", None)
            warmup_epochs: Warmup epochs for scheduler
            min_lr: Minimum learning rate for scheduler
        """
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.scheduler_type = scheduler
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.freeze_vision = freeze_vision
        self.embed_dim = embed_dim

        # Build location encoder with modular activation
        encoding_config = encoding_config or {}
        activation_config = activation_config or {}

        self.location_encoder = LocationEncoder(
            encoding_type=encoding_type,
            encoding_config=encoding_config,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            num_layers=num_layers,
            activation_type=activation_type,
            activation_config=activation_config,
        )

        # Build vision encoder
        self.vision_encoder = self._build_vision_encoder(vision_encoder, embed_dim)

        # Learnable temperature (log scale for stability)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def _build_vision_encoder(self, encoder_type: str, embed_dim: int) -> nn.Module:
        """Build vision encoder with MoCo pretrained weights.

        Args:
            encoder_type: Type of vision encoder
            embed_dim: Output embedding dimension

        Returns:
            Vision encoder module
        """
        if not HAS_TORCHGEO:
            print("Warning: torchgeo not installed. Using placeholder vision encoder.")
            print("Install with: pip install torchgeo timm")
            return self._placeholder_encoder(embed_dim)

        if encoder_type == "moco_resnet18":
            weights = ResNet18_Weights.SENTINEL2_ALL_MOCO
            in_chans = weights.meta["in_chans"]
            model = timm.create_model("resnet18", in_chans=in_chans, num_classes=embed_dim)
            model.load_state_dict(weights.get_state_dict(progress=True), strict=False)
            if self.freeze_vision:
                model.requires_grad_(False)
                model.fc.requires_grad_(True)
            return model

        elif encoder_type == "moco_resnet50":
            weights = ResNet50_Weights.SENTINEL2_ALL_MOCO
            in_chans = weights.meta["in_chans"]
            model = timm.create_model("resnet50", in_chans=in_chans, num_classes=embed_dim)
            model.load_state_dict(weights.get_state_dict(progress=True), strict=False)
            if self.freeze_vision:
                model.requires_grad_(False)
                model.fc.requires_grad_(True)
            return model

        elif encoder_type == "moco_vit16":
            weights = ViTSmall16_Weights.SENTINEL2_ALL_MOCO
            in_chans = weights.meta["in_chans"]
            model = timm.create_model("vit_small_patch16_224", in_chans=in_chans, num_classes=embed_dim)
            model.load_state_dict(weights.get_state_dict(progress=True), strict=False)
            if self.freeze_vision:
                model.requires_grad_(False)
                model.head.requires_grad_(True)
            return model

        else:
            raise ValueError(
                f"Unknown vision encoder: {encoder_type}. "
                "Options: moco_resnet18, moco_resnet50, moco_vit16"
            )

    def _placeholder_encoder(self, embed_dim: int) -> nn.Module:
        """Placeholder encoder for when torchgeo is not available."""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, embed_dim),
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to embedding.

        Args:
            image: (batch, C, H, W) image tensor. For Sentinel-2, C=13 channels.

        Returns:
            (batch, embed_dim) normalized image embeddings
        """
        return self.vision_encoder(image)

    def encode_location(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode location to embedding.

        Args:
            coords: (batch, 2) tensor with (lon, lat) in degrees

        Returns:
            (batch, embed_dim) normalized location embeddings
        """
        # Use float32 for compatibility with mixed precision training
        return self.location_encoder(coords.float())

    def forward(
        self, image: torch.Tensor, coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both encodings.

        Args:
            image: (batch, C, H, W) image tensor
            coords: (batch, 2) coordinate tensor

        Returns:
            Tuple of (image_features, location_features), both normalized
        """
        image_features = self.encode_image(image)
        location_features = self.encode_location(coords)

        # L2 normalize
        image_features = F.normalize(image_features, dim=1)
        location_features = F.normalize(location_features, dim=1)

        return image_features, location_features

    def contrastive_loss(
        self,
        image_features: torch.Tensor,
        location_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute symmetric contrastive loss (SatCLIP/CLIP loss).

        Args:
            image_features: Normalized image embeddings (batch, embed_dim)
            location_features: Normalized location embeddings (batch, embed_dim)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        logit_scale = self.logit_scale.exp().clamp(max=100)  # Clamp for stability

        # Compute similarity matrix
        logits_per_image = logit_scale * image_features @ location_features.t()
        logits_per_location = logits_per_image.t()

        # Labels are identity (diagonal should match)
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)

        # Symmetric cross entropy
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_l = F.cross_entropy(logits_per_location, labels)
        loss = (loss_i + loss_l) / 2

        # Compute accuracy for logging
        with torch.no_grad():
            acc_i = (logits_per_image.argmax(dim=1) == labels).float().mean()
            acc_l = (logits_per_location.argmax(dim=1) == labels).float().mean()

        metrics = {
            "loss_image": loss_i,
            "loss_location": loss_l,
            "acc_image": acc_i,
            "acc_location": acc_l,
            "logit_scale": logit_scale,
        }

        return loss, metrics

    def training_step(self, batch, batch_idx):
        """Training step."""
        images = batch["image"]
        coords = batch["point"].float()

        image_features, location_features = self(images, coords)
        loss, metrics = self.contrastive_loss(image_features, location_features)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_acc_img", metrics["acc_image"], prog_bar=True, sync_dist=True)
        self.log("train_acc_loc", metrics["acc_location"], sync_dist=True)
        self.log("logit_scale", metrics["logit_scale"], sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images = batch["image"]
        coords = batch["point"].float()

        image_features, location_features = self(images, coords)
        loss, metrics = self.contrastive_loss(image_features, location_features)

        # Log metrics
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc_img", metrics["acc_image"], prog_bar=True, sync_dist=True)
        self.log("val_acc_loc", metrics["acc_location"], sync_dist=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer with differential weight decay and optional scheduler."""
        # Separate params that should/shouldn't have weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or "bias" in name or "logit_scale" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
        )

        if self.scheduler_type is None:
            return optimizer

        # Add scheduler
        if self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.min_lr,
            )
        elif self.scheduler_type == "warmup_cosine":
            # Linear warmup then cosine decay
            def lr_lambda(epoch):
                if epoch < self.warmup_epochs:
                    return epoch / self.warmup_epochs
                progress = (epoch - self.warmup_epochs) / (self.trainer.max_epochs - self.warmup_epochs)
                return self.min_lr / self.learning_rate + (1 - self.min_lr / self.learning_rate) * (1 + np.cos(np.pi * progress)) / 2

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def get_location_encoder(self) -> LocationEncoder:
        """Get the trained location encoder for downstream use.

        Returns:
            The location encoder module
        """
        return self.location_encoder
