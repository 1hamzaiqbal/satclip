"""Globe visualization callback for TensorBoard logging.

Visualizes location encoder embeddings on a world map by sampling a grid
of locations, passing them through the encoder, and using PCA to color
the globe based on embedding similarity.

Inspired by TTE's Voronoi visualization approach.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

# Try to import cartopy for proper map projections
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def create_globe_grid(
    lat_resolution: int = 90,
    lon_resolution: int = 180,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a grid of lat/lon points covering the globe.

    Args:
        lat_resolution: Number of latitude points
        lon_resolution: Number of longitude points

    Returns:
        Tuple of (lats, lons, coords) where coords is (N, 2) array of (lon, lat)
    """
    lats = np.linspace(-90, 90, lat_resolution)
    lons = np.linspace(-180, 180, lon_resolution)

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Flatten and stack as (lon, lat) which is what the encoder expects
    coords = np.stack([lon_grid.flatten(), lat_grid.flatten()], axis=-1)

    return lats, lons, coords


class GlobeVisualizationCallback(Callback):
    """Callback to visualize location embeddings on a world map.

    Samples locations across the globe, encodes them, and uses PCA
    to create an RGB color mapping showing semantic similarity of
    different geographic regions.
    """

    def __init__(
        self,
        log_every_n_epochs: int = 10,
        log_on_train_start: bool = True,
        lat_resolution: int = 90,
        lon_resolution: int = 180,
        use_cartopy: bool = True,
    ):
        """Initialize callback.

        Args:
            log_every_n_epochs: Log globe visualization every N epochs
            log_on_train_start: Whether to log initial embeddings
            lat_resolution: Number of latitude points in grid
            lon_resolution: Number of longitude points in grid
            use_cartopy: Whether to use cartopy for map projection
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.log_on_train_start = log_on_train_start
        self.lat_resolution = lat_resolution
        self.lon_resolution = lon_resolution
        self.use_cartopy = use_cartopy and HAS_CARTOPY

        # Pre-compute grid
        self.lats, self.lons, self.coords = create_globe_grid(
            lat_resolution, lon_resolution
        )
        self.coords_tensor = torch.tensor(self.coords, dtype=torch.float32)

        if not HAS_SKLEARN:
            print("Warning: sklearn not installed. Globe visualization will be limited.")

    def _encode_globe(self, pl_module: pl.LightningModule) -> np.ndarray:
        """Encode all grid points using the location encoder.

        Args:
            pl_module: Lightning module with location_encoder

        Returns:
            Embeddings array of shape (N, embed_dim)
        """
        device = next(pl_module.parameters()).device
        coords = self.coords_tensor.to(device)

        # Encode in batches to avoid OOM
        batch_size = 4096
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(coords), batch_size):
                batch = coords[i:i + batch_size]
                emb = pl_module.encode_location(batch)
                embeddings.append(emb.cpu())

        return torch.cat(embeddings, dim=0).numpy()

    def _embeddings_to_colors(self, embeddings: np.ndarray) -> np.ndarray:
        """Convert embeddings to RGB colors using PCA.

        Args:
            embeddings: (N, embed_dim) array

        Returns:
            (N, 3) RGB colors in [0, 1]
        """
        if not HAS_SKLEARN:
            # Fallback: use first 3 dimensions
            colors = embeddings[:, :3].copy()
        else:
            # PCA to 3 dimensions
            pca = PCA(n_components=3)
            colors = pca.fit_transform(embeddings)

        # Normalize to [0, 1]
        colors = colors - colors.min(axis=0)
        colors = colors / (colors.max(axis=0) + 1e-8)

        return colors

    def _create_globe_figure_cartopy(
        self,
        colors: np.ndarray,
        epoch: int,
    ) -> plt.Figure:
        """Create globe visualization using cartopy.

        Args:
            colors: (N, 3) RGB colors
            epoch: Current epoch

        Returns:
            Matplotlib figure
        """
        # Reshape colors to grid
        color_grid = colors.reshape(self.lat_resolution, self.lon_resolution, 3)

        fig = plt.figure(figsize=(14, 7))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

        ax.set_global()

        # Plot as colored scatter (more reliable than pcolormesh with projections)
        lat_grid, lon_grid = np.meshgrid(self.lats, self.lons, indexing='ij')

        ax.scatter(
            lon_grid.flatten(),
            lat_grid.flatten(),
            c=colors,
            s=3,
            alpha=0.8,
            transform=ccrs.PlateCarree(),
            marker='s',
        )

        # Add map features on top
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5, zorder=3)

        ax.set_title(f'Location Embeddings (PCA → RGB)\nEpoch {epoch}', fontsize=14)

        plt.tight_layout()
        return fig

    def _create_globe_figure_basic(
        self,
        colors: np.ndarray,
        epoch: int,
    ) -> plt.Figure:
        """Create globe visualization using basic matplotlib.

        Args:
            colors: (N, 3) RGB colors
            epoch: Current epoch

        Returns:
            Matplotlib figure
        """
        # Reshape colors to grid
        color_grid = colors.reshape(self.lat_resolution, self.lon_resolution, 3)

        fig, ax = plt.subplots(figsize=(14, 7))

        # Use imshow for grid visualization
        extent = [-180, 180, -90, 90]
        ax.imshow(
            color_grid,
            extent=extent,
            origin='lower',
            aspect='equal',
        )

        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Location Embeddings (PCA → RGB)\nEpoch {epoch}', fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _log_globe(self, trainer: pl.Trainer, pl_module: pl.LightningModule, epoch: int):
        """Log globe visualization to TensorBoard.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
            epoch: Current epoch
        """
        # Check if module has location encoder
        if not hasattr(pl_module, 'encode_location'):
            return

        # Encode globe
        embeddings = self._encode_globe(pl_module)

        # Convert to colors
        colors = self._embeddings_to_colors(embeddings)

        # Create figure
        if self.use_cartopy:
            fig = self._create_globe_figure_cartopy(colors, epoch)
        else:
            fig = self._create_globe_figure_basic(colors, epoch)

        # Log to TensorBoard
        if trainer.logger is not None:
            trainer.logger.experiment.add_figure(
                'globe/embeddings',
                fig,
                global_step=epoch,
            )

        plt.close(fig)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Log initial globe visualization."""
        if self.log_on_train_start:
            self._log_globe(trainer, pl_module, epoch=0)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Log globe visualization at end of epoch."""
        current_epoch = trainer.current_epoch

        if (current_epoch + 1) % self.log_every_n_epochs == 0:
            self._log_globe(trainer, pl_module, epoch=current_epoch + 1)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Log final globe visualization."""
        current_epoch = trainer.current_epoch
        if (current_epoch + 1) % self.log_every_n_epochs != 0:
            self._log_globe(trainer, pl_module, epoch=current_epoch + 1)


class CombinedVisualizationCallback(Callback):
    """Callback that combines spline and globe visualizations side by side.

    Creates a dashboard-style figure showing:
    - Left: Spline activation shapes (if using spline activation)
    - Right: Globe embeddings colored by PCA
    """

    def __init__(
        self,
        log_every_n_epochs: int = 10,
        log_on_train_start: bool = True,
        lat_resolution: int = 60,
        lon_resolution: int = 120,
    ):
        """Initialize callback.

        Args:
            log_every_n_epochs: Log visualization every N epochs
            log_on_train_start: Whether to log initial state
            lat_resolution: Number of latitude points
            lon_resolution: Number of longitude points
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.log_on_train_start = log_on_train_start

        # Initialize sub-components
        self.lats, self.lons, self.coords = create_globe_grid(
            lat_resolution, lon_resolution
        )
        self.lat_resolution = lat_resolution
        self.lon_resolution = lon_resolution
        self.coords_tensor = torch.tensor(self.coords, dtype=torch.float32)

    def _find_spline_activations(self, model):
        """Find all SplineActivation modules."""
        from experiments.models.activations.spline import SplineActivation
        splines = []
        for name, module in model.named_modules():
            if isinstance(module, SplineActivation):
                splines.append((name, module))
        return splines

    def _encode_globe(self, pl_module):
        """Encode globe grid."""
        device = next(pl_module.parameters()).device
        coords = self.coords_tensor.to(device)

        batch_size = 4096
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(coords), batch_size):
                batch = coords[i:i + batch_size]
                emb = pl_module.encode_location(batch)
                embeddings.append(emb.cpu())

        return torch.cat(embeddings, dim=0).numpy()

    def _create_combined_figure(
        self,
        pl_module: pl.LightningModule,
        epoch: int,
    ) -> plt.Figure:
        """Create combined dashboard figure.

        Args:
            pl_module: Lightning module
            epoch: Current epoch

        Returns:
            Matplotlib figure
        """
        splines = self._find_spline_activations(pl_module)
        has_splines = len(splines) > 0

        # Figure layout
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
            colors = plt.cm.viridis(np.linspace(0, 1, len(splines)))

            for (name, spline), color in zip(splines, colors):
                knot_x, knot_y = spline.get_knot_data()
                with torch.no_grad():
                    y = spline(x.to(knot_x.device)).cpu()

                short_name = name.split('.')[-2] if '.' in name else name
                ax_spline.plot(x.numpy(), y.numpy(), color=color, linewidth=2, label=short_name)
                ax_spline.scatter(knot_x.numpy(), knot_y.numpy(), c=[color], s=20, zorder=5)

            # Reference
            ax_spline.plot(x.numpy(), np.maximum(0, x.numpy()), 'k--', alpha=0.3, label='ReLU')

            ax_spline.set_xlabel('Input')
            ax_spline.set_ylabel('Output')
            ax_spline.set_title(f'Spline Activations')
            ax_spline.legend(fontsize=8)
            ax_spline.grid(True, alpha=0.3)
            ax_spline.set_xlim(-4, 4)

        # Plot globe
        embeddings = self._encode_globe(pl_module)

        if HAS_SKLEARN:
            pca = PCA(n_components=3)
            colors = pca.fit_transform(embeddings)
        else:
            colors = embeddings[:, :3]

        colors = colors - colors.min(axis=0)
        colors = colors / (colors.max(axis=0) + 1e-8)

        lat_grid, lon_grid = np.meshgrid(self.lats, self.lons, indexing='ij')

        if HAS_CARTOPY and isinstance(ax_globe, plt.Axes):
            # Check if it's a GeoAxes
            if hasattr(ax_globe, 'set_global'):
                ax_globe.set_global()
                ax_globe.scatter(
                    lon_grid.flatten(), lat_grid.flatten(),
                    c=colors, s=2, alpha=0.8,
                    transform=ccrs.PlateCarree(), marker='s',
                )
                ax_globe.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
            else:
                # Fallback
                color_grid = colors.reshape(self.lat_resolution, self.lon_resolution, 3)
                ax_globe.imshow(color_grid, extent=[-180, 180, -90, 90], origin='lower')
        else:
            color_grid = colors.reshape(self.lat_resolution, self.lon_resolution, 3)
            ax_globe.imshow(color_grid, extent=[-180, 180, -90, 90], origin='lower', aspect='equal')
            ax_globe.set_xlabel('Longitude')
            ax_globe.set_ylabel('Latitude')

        ax_globe.set_title(f'Location Embeddings (PCA → RGB)')

        fig.suptitle(f'Training Progress - Epoch {epoch}', fontsize=14, y=1.02)
        plt.tight_layout()

        return fig

    def _log_visualization(self, trainer: pl.Trainer, pl_module: pl.LightningModule, epoch: int):
        """Log combined visualization."""
        if not hasattr(pl_module, 'encode_location'):
            return

        fig = self._create_combined_figure(pl_module, epoch)

        if trainer.logger is not None:
            trainer.logger.experiment.add_figure(
                'dashboard/combined',
                fig,
                global_step=epoch,
            )

        plt.close(fig)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.log_on_train_start:
            self._log_visualization(trainer, pl_module, epoch=0)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        current_epoch = trainer.current_epoch
        if (current_epoch + 1) % self.log_every_n_epochs == 0:
            self._log_visualization(trainer, pl_module, epoch=current_epoch + 1)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        current_epoch = trainer.current_epoch
        if (current_epoch + 1) % self.log_every_n_epochs != 0:
            self._log_visualization(trainer, pl_module, epoch=current_epoch + 1)
